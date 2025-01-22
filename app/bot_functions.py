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
from app.mg_functions import omega_ratio
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
    density = df['cross'].sum()/len(df)
    #df['strategy'] = (1+df['return']).cumprod() - 1
    return df, density


def calmar_ratio(returns, periods_per_year=252):
    avg_return = np.mean(returns)
    annualized_return = (1 + avg_return) ** periods_per_year - 1
    cumulative_returns = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = running_max - cumulative_returns
    max_drawdown = np.max(drawdowns)
    return annualized_return / max_drawdown if max_drawdown > 0 else float('inf')


def garch_metric(excess_returns):
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    cumulative_returns = excess_returns.cumsum()
    growth_continuity = np.sum(cumulative_returns.diff() < 0) / len(cumulative_returns) if len(cumulative_returns) > 1 else 0
    final_result = 1 if cumulative_returns.iloc[-1] > 0 else 0
    garch_metric_value = sharpe_ratio * (1 - growth_continuity) * final_result
    return garch_metric_value


def wlr_rr(df_raw):
    df = df_raw.copy()
    df['stance'] = df['stance'].shift(1)
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    stances = df['stance'].to_numpy()
    positions = []
    for index, position in enumerate(stances):
        last_position = stances[index-1] if index > 0 else None
        if position != last_position:
           positions.append((index, position))
    positions = [(a[0], b[0], int(a[1])) for a, b in zip(positions[1:], positions[2:])]
    stats = []
    for start, end, position in positions:
        df_stats = df.iloc[start:end]
        prices = df_stats[['close', 'high', 'low']].to_numpy()
        #print(prices)
        open_price, close_price = prices[0, 0], prices[-1, 0]
        max_price, min_price = prices[:, 1].max(), prices[:, 2].min()

        if position == 1:
            result = (close_price - open_price) / open_price
            max_result = (max_price - open_price) / open_price
            min_result = (open_price - min_price) / open_price
        else:
            result = (open_price - close_price) / open_price
            min_result = (max_price - open_price) / open_price
            max_result = (open_price - min_price) / open_price

        stats.append((result, min_result, max_result))
    # Tworzenie DataFrame dla statystyk
    stats_df = pd.DataFrame(stats, columns=['result', 'min_result', 'max_result'])

    percent_best = 3
    number = round((percent_best/100)*len(stats_df))
    max_res = stats_df.sort_values(by=['max_result'], ascending=False)['max_result'][number:]
    aberration = max_res.mean() + stats_df['result'].std()
    stats_df['result'] = np.where(stats_df['result'] > aberration, aberration, stats_df['result'])

    risk_reward_ratio = stats_df['max_result'].mean() / (stats_df['min_result'].mean() + stats_df['min_result'].std())
    risk_reward_ratio = round(risk_reward_ratio, 3)
    try:
        win_loss_ratio = round(len(stats_df[stats_df['result'] > 0])/len(stats_df[stats_df['result'] < 0]), 3)
    except ZeroDivisionError:
        win_loss_ratio = 1
    end_result = round(risk_reward_ratio * win_loss_ratio, 2)
    if (end_result == np.inf) or (end_result > 2):
        end_result = 2.0

    garch = garch_metric(stats_df['result'])
    return round(end_result*garch, 5)


def calc_result(df, sharpe_multiplier, check_week_ago=False, check_end_result=False):
    if check_week_ago:
        today = dt.now().date()
        week_ago_date = today - timedelta(days=7)
        df = df[(df['date_xy'] >= week_ago_date)]

    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    cross = int(df['cross'].sum()) ** 0.85 + 2
    sharpe = round(sharpe_multiplier*((df['return'].mean()/df['return'].std()))/cross, 6)
    omega = omega_ratio(df['return'])
    if check_end_result:
        end_result = wlr_rr(df)
        return sharpe, omega, end_result
    else:
        return sharpe, omega


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


def win_ratio(df, column, window=50, threshold=0):
    """
    :Oblicza wskaźnik Omega w sposób rolling dla określonej kolumny DataFrame.
    :param df: DataFrame zawierający dane.
    :param column: Nazwa kolumny ze stopami zwrotu.
    :param window: Wielkość kroczącego okna (domyślnie 200).
    :param threshold: Minimalna oczekiwana stopa zwrotu (domyślnie 0).
    :return: DataFrame z nową kolumną zawierającą rolling Omega ratio.
    """
    def win_ratio_(returns, threshold):
        # Liczymy liczniki i mianowniki wskaźnika Omega
        gains = np.sum(returns[returns > threshold] - threshold)
        losses = np.sum(threshold - returns[returns <= threshold])
        win_count = np.sum(returns > threshold)
        loss_count = np.sum(returns <= threshold)
        return (win_count / loss_count) * (gains / losses) if (losses != 0)&(loss_count!=0) else np.nan  # Unikamy dzielenia przez zero

    # Obliczanie rolling Omega ratio
    df['win_ratio'] = df[column].rolling(window).apply(
        lambda x: win_ratio_(x, threshold), raw=True
    )
    df['win_ratio_fast'] = df['win_ratio'].rolling(round(window/10)).mean()
    df['win_ratio_slow'] = df['win_ratio'].rolling(window).mean()
    return df


def find_support_resistance_numpy(df, slow, fast):
    df['tolerance'] = (df.high-df.low).rolling(slow).mean() * fast/20
    low = df['low'].to_numpy()
    high = df['high'].to_numpy()
    tolerance = df['tolerance'].to_numpy()

    # Znajdź lokalne minima i maksima za pomocą NumPy
    is_support = np.where((low[1:-1] < low[:-2] - tolerance[1:-1]) & (low[1:-1] < low[2:] - tolerance[1:-1]))[0] + 1
    is_resistance = np.where((high[1:-1] > high[:-2] + tolerance[1:-1]) & (high[1:-1] > high[2:] + tolerance[1:-1]))[0] + 1

    # Utwórz kolumny support i resistance
    df['support'] = np.nan
    df['resistance'] = np.nan
    df.loc[df.index[is_support], 'support'] = df['low'].iloc[is_support]
    df.loc[df.index[is_resistance], 'resistance'] = df['high'].iloc[is_resistance]

    # Wypełnij brakujące wartości NaN poprzednimi wartościami (forward fill)
    df['support'] = df['support'].ffill()
    df['resistance'] = df['resistance'].ffill()
    return df


def play_with_trend(symbol, short, long, dev):
    df = get_data(symbol, 'M5', 1, 5000)
    df['ma_week'] = df.ta.sma(length=long)
    df['ma_432'] = df.ta.sma(length=short)
    df['std_432'] = df['close'].rolling(short).std()
    df['boll_up_432'] = df['ma_432'] + (dev/10)*df['std_432']
    df['boll_down_432'] = df['ma_432'] - (dev/10)*df['std_432']
    d = df.iloc[-1]
    if d.ma_432 > d.ma_week:
        if d.close < d.boll_down_432:
            print("Best to open long!")
            return 0.55
        elif d.close < d.ma_432:
            print("Good to open long!")
            return 0.35
        elif d.close > d.ma_432 and d.close < d.boll_up_432:
            print("Just long trend!")
            return 0.2
    else:
        if d.close > d.boll_up_432:
            print("Best to open short!")
            return -0.55
        elif d.close > d.ma_432:
            print("Good to open short!")
            return -0.35
        elif d.close < d.ma_432 and d.close > d.boll_down_432:
            print("Just short trend!")
            return -0.2
    return 0


def play_with_trend_bt(symbol):
    from tqdm import tqdm
    longs = range(1100, 2001, 25)
    shorts = range(380, 721, 10)
    devs = range(10, 23, 2)
    df_raw = get_data(symbol, 'M5', 1, 50000)

    def trend_strategy(short, long, dev):
        df = df_raw.copy()
        df['ma_short'] = df.ta.ema(length=short)
        df['ma_long'] = df.ta.ema(length=long)
        df['stddev'] = df.close.rolling(short).std()
        df['boll_up'] = df['ma_short'] + (dev/10)*df['stddev']
        df['boll_down'] = df['ma_short']-(dev/10)*df['stddev']
        up = df.ma_short > df.ma_long
        down = df.ma_short < df.ma_long
        df['stance'] = 0
        close_long = np.where((up)&(df.close.shift()>df.boll_up.shift())&(df.close<df.boll_up), 3, 0)
        stance_up_1 = np.where((up)&(df.close.shift()>df.ma_short.shift())&(df.close<df.ma_short), 1, 0)
        stance_up_2 = np.where((up)&(df.close.shift()>df.boll_down.shift())&(df.close<df.boll_down), 2, 0)

        close_short = np.where((down)&(df.close.shift()<df.boll_down.shift())&(df.close>df.boll_down), -3, 0)
        stance_down_1 = np.where((down)&(df.close.shift()<df.ma_short.shift())&(df.close>df.ma_short), -1, 0)
        stance_down_2 = np.where((down)&(df.close.shift()<df.boll_up.shift())&(df.close>df.boll_up), -2, 0)
        df['stance'] = close_long + stance_up_1 + stance_up_2 + close_short + stance_down_1 + stance_down_2
        df['stance'] = df['stance'].replace(0, np.NaN)
        df['stance'] = df['stance'].replace([-3, 3], 0)
        df['stance'] = df['stance'].ffill()
        return df

    results = []
    for long in tqdm(longs):
        for short in shorts:
            if short == long or short>long:
                continue
            for dev in devs:
                try:
                    df = trend_strategy(short, long, dev)
                    df, _ = calculate_strategy_returns(df, 1)
                    df = df.dropna()
                    df.reset_index(drop=True, inplace=True)
                    sharpe = round(((df['return'].mean()/df['return'].std())), 6)
                    omega = omega_ratio(df['return'])
                    df2 = df.copy()[int(len(df)/2):]
                    sharpe2 = round(((df2['return'].mean()/df2['return'].std())), 6)
                    omega2 = omega_ratio(df2['return'])
                    result = np.mean([sharpe, sharpe2])*np.mean([omega, omega2])
                    results.append((dev, short, long, result))
                except Exception as e:
                    print(e)
                    continue

    final = sorted(results, key=lambda x: x[3], reverse=True)[0]
    dev = final[0]
    short = final[1]
    long = final[2]
    print(f'play_with_trend_bt results: dev={dev}, short={short}, long={long}, result={round(final[3],5)}')
    return short, long, dev

