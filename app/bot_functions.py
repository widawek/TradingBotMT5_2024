import os
from datetime import datetime as dt
import pandas_ta as ta
from datetime import timedelta
import MetaTrader5 as mt
import pandas as pd
import numpy as np
import sys
sys.path.append("..")
from app.functions import get_data
mt.initialize()


def rename_files_in_directory(old_phrase, new_phrase, catalog):
    # Iterate over all files in the specified directory
    for filename in os.listdir(catalog):
        # Check if the old phrase is in the filename
        if old_phrase in filename:
            # Create the new filename by replacing the old phrase with the new one
            new_filename = filename.replace(old_phrase, new_phrase)
            # Construct the full old and new file paths
            old_file_path = os.path.join(catalog, filename)
            new_file_path = os.path.join(catalog, new_filename)

            def change_(old_file_path, new_file_path):
                os.rename(old_file_path, new_file_path)
                #print(f'Renamed: {old_file_path} -> {new_file_path}')
            # Rename the file
            try:
                change_(old_file_path, new_file_path)
            except FileExistsError:
                new_file_path = new_file_path.split('_')
                result = int(new_file_path[-2])
                new_file_path[-2] = str(result + 1)
                new_file_path = '_'.join(new_file_path)
                change_(old_file_path, new_file_path)


def checkout_report(symbol, reverse, trigger, condition):
    from_date = dt.today().date() - timedelta(days=0)
    print(from_date)
    to_date = dt.today().date() + timedelta(days=1)
    print(f"Data from {from_date.strftime('%A')} {from_date} to {to_date.strftime('%A')} {to_date}")
    from_date = dt(from_date.year, from_date.month, from_date.day)
    to_date = dt(to_date.year, to_date.month, to_date.day)
    try:
        data = mt.history_deals_get(from_date, to_date)
        df = pd.DataFrame(list(data), columns=data[0]._asdict().keys())
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df[(df['symbol'] != '') & (df['symbol']==symbol)]
        df['profit'] = df['profit'] + df['commission'] * 2 + df['swap']
        df['reason'] = df['reason'].shift(1)
        df = df.drop(columns=['time_msc', 'commission', 'external_id', 'swap'])
        df = df[df['time'].dt.date >= from_date.date()]
        df = df[df.groupby('position_id')['position_id'].transform('count') == 2]
        df = df.sort_values(by=['position_id', 'time'])
        df.reset_index(drop=True, inplace=True)
        df['profit'] = df['profit'].shift(-1)
        df['time_close'] = df['time'].shift(-1)
        df = df.rename(columns={'time': 'time_open'})
        df['sl'] = df.comment.shift(-1).str.contains('sl', na=False)
        df['tp'] = df.comment.shift(-1).str.contains('tp', na=False)
        df = df.iloc[::2]

        if len(df) < 3:
            return False

        prof_lst = df['profit'][-3:].to_list()
        comm_lst = df['comment'][-3:].to_list()
        if comm_lst[0][0] == reverse[0]:
            if comm_lst[0][2] == trigger[-1]:

                # BotReverse
                if condition:
                    return all([all([i>0 for i in prof_lst]),
                            all([i==comm_lst[0] for i in comm_lst])])

                return all([all([i<0 for i in prof_lst]),
                            all([i==comm_lst[0] for i in comm_lst])])
    except Exception as e:
        print("checkout_report", e)
        return False

    return False


def vwap_std(symbol, factor=1.4):
    daily = get_data(symbol, 'D1', 0, 10)
    daily = daily.drop(columns=['open', 'close', 'spread', 'volume'])
    daily['high'] = daily['high'].shift(1)
    daily['low'] = daily['low'].shift(1)
    daily['date'] = pd.to_datetime(daily.time.dt.date)
    daily.dropna(inplace=True)

    df = get_data(symbol, 'M1', 1, 1440)
    df['date'] = pd.to_datetime(df.time.dt.date)
    df = df.reset_index()
    dates = list(set(df.date.to_list()))
    dates.sort()
    opens = []
    intraday_mean = []
    std = []
    high_ = []
    low_ = []
    for i in dates:
        daily_df_x = daily[daily['date'] == i]
        high_value = float(daily_df_x['high'].iloc[-1])
        low_value = float(daily_df_x['low'].iloc[-1])

        x = df.copy()[df['date'] == i]
        x['max'] = x.high.rolling(window=len(x), min_periods=0).max()
        x['min'] = x.low.rolling(window=len(x), min_periods=0).min()
        x['cv'] = x['close'] * x['volume']
        x['cv'] = x['cv'].rolling(window=len(x), min_periods=0).sum()
        x['vol'] = x.volume.rolling(window=len(x), min_periods=0).sum()
        x['mean'] = x['cv']/x['vol']
        x['std_'] = x.close.rolling(window=len(x), min_periods=0).std()
        x['highs1'] = x['high'] - x['high'].shift(1)
        x['lows1'] = x['low'].shift(1) - x['low']
        x['highs_'] = np.where(x['highs1']>0, x['highs1'], 0)
        x['lows_'] = np.where(x['lows1']>0, x['lows1'], 0)
        x['highss'] = x['highs_'].ewm(min_periods=0, alpha=1/len(x)).mean()
        x['lowsss'] = x['lows_'].ewm(min_periods=0, alpha=1/len(x)).mean()
        x.loc[:,'open'] = x['open'].iloc[0]
        x.loc[:,'last_high'] = high_value
        x.loc[:,'last_low'] = low_value
        opens += x['open'].to_list()
        intraday_mean += x['mean'].to_list()
        std += x['std_'].to_list()
        high_ += x['last_high'].to_list()
        low_ += x['last_low'].to_list()

    df['open'] = pd.Series(opens)
    df['daily_mean'] = pd.Series(intraday_mean)
    df['std_'] = pd.Series(std)
    df['boll_up'] = df['daily_mean'] + factor*df['std_']
    df['boll_down'] = df['daily_mean'] - factor*df['std_']
    df['last_daily_high'] = pd.Series(high_)
    df['last_daily_low'] = pd.Series(low_)
    df['boll_up_max'] = np.where((df['boll_up']>df['last_daily_high']) &
                                 (df['boll_up'].shift(1)<df['last_daily_high'].shift(1)), 1, 0)
    df['boll_down_min'] = np.where((df['boll_down']<df['last_daily_low']) &
                                 (df['boll_down'].shift(1)>df['last_daily_low'].shift(1)), 1, 0)
    df['boll_up_max'] = df['boll_up_max'].rolling(5).max()
    df['boll_down_min'] = df['boll_down_min'].rolling(5).max()

    last_open = df['open'].iloc[-1]
    last_mean = df['daily_mean'].iloc[-1]
    last_boll_up = df['boll_up'].iloc[-1]
    last_boll_down = df['boll_down'].iloc[-1]
    last_close = df['close'].iloc[-1]
    last_high = df['last_daily_high'].iloc[-1]
    last_low = df['last_daily_low'].iloc[-1]

    last_min = df['low'][-3:-1].min()
    last_max = df['high'][-3:-1].max()
    last_boll_up_max = df['boll_up_max'].iloc[-1]
    last_boll_down_min = df['boll_down_min'].iloc[-1]

    #vwap trend 'long'
    if last_mean > last_open:
        if last_close > last_boll_up:
            trend = 'long_strong'
        elif last_close < last_boll_down and last_min < last_low:
            trend = 'long_super_weak'
        elif last_close < last_boll_down:
            trend = 'long_weak'
        else:
            trend = 'long_normal'
    #vwap trend 'short'
    elif last_mean < last_open:
        if last_close < last_boll_down:
            trend = 'short_strong'
        elif last_close > last_boll_up and last_max > last_high:
            trend = 'short_super_weak'
        elif last_close > last_boll_up:
            trend = 'short_weak'
        else:
            trend = 'short_normal'
    else:
        trend = 'neutral'

    if last_boll_up_max and last_max > last_high:
        trend = 'overbought'
    elif last_boll_down_min and last_min < last_low:
        trend = 'sold_out'
    return trend


def avg_daily_vol_for_divider(symbol: str, base: int) -> int:
    df1 = get_data(symbol, 'D1', 2, 30)
    df2 = get_data(symbol, 'D1', 1, 1)
    df1['avg_daily'] = (df1.high - df1.low) / df1.open
    df2['avg_daily'] = (df2.high - df2.low) / df2.open
    factor = df1['avg_daily'].mean()/df2['avg_daily'].mean()
    factor = 1+(factor-1)/2
    return int(round(base*factor))


def trend_or_not(symbol):
    factor = 15
    df = get_data(symbol, 'D1', 1, 100)
    stoch = df.ta.stoch(fast_k=factor, slow_k=factor, slow_d=factor)
    df['pct_change'] = (df['close'] - df['close'].shift(factor)) / df['close'].shift(factor)
    df['k'] = stoch.iloc[:,0] * df['pct_change']
    df['d'] = stoch.iloc[:,1] * df['pct_change']
    df = df.dropna()
    if df['k'].iloc[-1] > df['d'].iloc[-1]:
        print("Trend!")
        return True
    print("Not trend!")
    return False


def close_request_only(position):
    request = {"action": mt.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": float(position.volume),
            "type": 1 if (position.type == 0) else 0,
            "position": position.ticket,
            "magic": position.magic,
            'deviation': 20,
            "type_time": mt.ORDER_TIME_GTC,
            "type_filling": mt.ORDER_FILLING_IOC
            }
    order_result = mt.order_send(request)
    print(order_result)


def close_request_(symbol: str, tickets_list, only_profitable: bool=False):
    if symbol == "ALL":
        positions_ = mt.positions_get()
    else:
        positions_ = mt.positions_get(symbol=symbol)
    for i in positions_ :
        if only_profitable:
            if i.profit <= 0:
                continue
        if i.ticket in tickets_list:
            close_request_only(i)


def calculate_strategy_returns(df, leverage):
    """The dataframe has to have a 'stance' column."""
    z = [len(str(x).split(".")[1])+1 for x in list(df["close"][:101])]
    divider = 10**round((sum(z)/len(z))-1)
    spread_mean = df.spread/divider
    spread_mean = spread_mean.mean()
    df["cross"] = np.where( ((df.stance == 1) & (df.stance.shift(1) != 1)) | \
                            ((df.stance == -1) & (df.stance.shift(1) != -1)), 1, 0 )
    df['mkt_move'] = np.log(df.close/df.close.shift(1))
    df['return'] = (df.mkt_move * df.stance.shift(1) - (df["cross"] *(spread_mean)/df.open))*leverage
    #df['strategy'] = (1+df['return']).cumprod() - 1
    return df

def calmar_ratio(returns, periods_per_year=252):
    avg_return = np.mean(returns)
    annualized_return = (1 + avg_return) ** periods_per_year - 1
    cumulative_returns = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = running_max - cumulative_returns
    max_drawdown = np.max(drawdowns)
    return annualized_return / max_drawdown if max_drawdown > 0 else float('inf')

def calc_result(df, sharpe_multiplier, check_week_ago=False):
    if check_week_ago:
        today = dt.now().date()
        week_ago_date = dt.now().date() - timedelta(days=7)
        #two_weeks_ago_date = dt.now().date() - timedelta(days=14)
        
        df = df[(df['date'] == today)|
                (df['date'] == week_ago_date)]#|
                #(df['date'] == two_weeks_ago_date)]

    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    cross = df['cross'].sum()/len(df)
    sharpe = round(sharpe_multiplier*((df['return'].mean()/df['return'].std()))/cross, 2)
    
    if not check_week_ago:
        calmar = calmar_ratio(df['return'])
    else:
        calmar = 0
    return sharpe, calmar

def delete_last_day_and_clean_returns(df, morning_hour, evening_hour, respect_overnight=True):
    df = df.dropna()
    df['date_xy'] = df['time'].dt.date
    x = list(set(np.unique(df['date_xy'])))
    x.sort()
    df = df[df['date_xy'] != x[0]]
    if not respect_overnight:
        df['return'] = np.where((df['time'].dt.hour < morning_hour-1) | (df['time'].dt.hour > evening_hour+1), np.NaN, df['return'])
        df = df.dropna()
    df['return'] = np.where(df['date_xy'] != df['date_xy'].shift(1), 0, df['return'])
    df.reset_index(drop=True, inplace=True)
    return df

def calculate_bars_to_past(df):
    df_dates = df.copy()
    df_dates['date'] = df_dates['time'].dt.date
    if any([True for i in np.unique(df_dates['date']) if i.weekday() in [5,6]]):
        small_bt_bars = 12000
    else:
        small_bt_bars = 10000
    return small_bt_bars
