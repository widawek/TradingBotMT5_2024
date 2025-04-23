import os
from datetime import datetime as dt
import pandas_ta as ta
from datetime import timedelta
import MetaTrader5 as mt
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
from math import ceil
import sys
import json
sys.path.append("..")
import copy
from app.functions import get_data
from strategies_analize.metrics import *
from scipy.stats import linregress
#from app.mg_functions import omega_ratio
mt.initialize()


def sharpe_ratio(returns, risk_free_rate=0):
    """Sharpe ratio = (średni zwrot - stopa wolna od ryzyka) / odchylenie standardowe zwrotów"""
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()


def omega_ratio(returns, threshold=0):
    """Omega ratio = suma zwrotów powyżej threshold / suma zwrotów poniżej threshold"""
    gains = returns[returns > threshold].sum()
    losses = abs(returns[returns < threshold].sum())
    return gains / losses if losses != 0 else 1.2


def max_drawdown(strategy):
    """Max Drawdown = największe obsunięcie kapitału"""
    peak = strategy.cummax()
    drawdown = (strategy - peak) / peak
    return drawdown.min()/100


def ulcer_index(strategy):
    """Ulcer Index = średnia kwadratowa drawdownów"""
    peak = strategy.cummax()
    drawdown = (strategy - peak) / peak
    return np.sqrt(np.mean(drawdown**2))


def cagr(strategy, years):
    """CAGR = skumulowana roczna stopa zwrotu"""
    return (strategy.iloc[-1] / strategy.iloc[0])**(1/years) - 1


def equity_curve_stability(strategy):
    """R^2 dopasowania equity curve do regresji liniowej"""
    x = np.arange(len(strategy))
    log_strategy = np.log(strategy)
    slope, intercept, r_value, _, _ = linregress(x, log_strategy)
    return r_value**2


def strategy_score(returns, sharpe_multiplier, years=1):
    """Finalna metryka"""
    years=1
    sharpe = sharpe_multiplier*sharpe_ratio(returns)
    omega = omega_ratio(returns)
    strategy = (1+returns).cumprod()
    #mdd = abs(max_drawdown(strategy))
    ui = ulcer_index(strategy)
    cagr_value = cagr(strategy, years)
    stability = equity_curve_stability(strategy)
    return (sharpe * omega * cagr_value * stability) / ui#*mdd)


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
    z = [len(str(x).split(".")[1])+1 for x in list(df["close"][-101:])]
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


def garch_metric(excess_returns):
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    cumulative_returns = excess_returns.cumsum()
    growth_continuity = np.sum(cumulative_returns.diff() < 0) / len(cumulative_returns) if len(cumulative_returns) > 1 else 0
    final_result = 1 if cumulative_returns.iloc[-1] > 0 else 0
    garch_metric_value = sharpe_ratio * (1 - growth_continuity) * final_result
    return garch_metric_value


def monte_carlo_with_shuffle(returns, num_trials=200, dropout_rate=0.1):

    def remove_lowest_10_percent(lst):
        if not lst:
            return lst
        lst = lst.tolist()
        positive_values = sorted([x for x in lst if x > 0], key=abs)
        n = ceil(len(positive_values) * 0.1)
        filtered_positives = positive_values[n:]
        negative_values = [x for x in lst if x <= 0]
        new_lst = negative_values + filtered_positives
        return pd.Series(new_lst)

    def calc_base(returns_):
        peak = 1
        max_drawdown = 0
        cumulative_returns = (1 + returns_).cumprod()
        sharpe = returns_.mean()/returns_.std()
        for value in cumulative_returns:
            peak = max(peak, value)
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        final_return = cumulative_returns.iloc[-1]
        return (final_return/max_drawdown)*sharpe

    final_ = calc_base(returns)
    #returns = remove_lowest_10_percent(returns)

    #results_to_plot = []
    #results_to_plot.append((1 + returns).cumprod())
    results = []
    for _ in range(num_trials):
        shuffled_returns = returns.sample(frac=1, replace=False).values
        num_to_drop = int(len(shuffled_returns) * dropout_rate)
        indices_to_drop = np.random.choice(len(shuffled_returns), num_to_drop, replace=False)
        modified_returns = np.delete(shuffled_returns, indices_to_drop)
        cumulative_returns = (1 + modified_returns).cumprod()
        #results_to_plot.append(pd.Series(cumulative_returns))
        peak = 1
        max_drawdown = 0
        for value in cumulative_returns:
            peak = max(peak, value)
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        final_return = cumulative_returns[-1] if len(cumulative_returns) > 0 else 0
        sharpe = modified_returns.mean()/modified_returns.std()
        results.append([final_return, max_drawdown, sharpe])
    # plt.figure(figsize=(10, 6))
    # # --- Rysowanie pierwszej serii (pogrubiona i na wierzchu) ---
    # plt.plot(results_to_plot[0].values, linewidth=3, zorder=2)
    # # Rysowanie pozostałych serii
    # for i, series in enumerate(results_to_plot[1:]):  # Zaczynamy od 1, nie od 0
    #     plt.plot(series.values, label=f'Seria {i+2}', zorder=1)

    # plt.title('Wykres serii o różnych długościach')
    # plt.grid(True)
    # plt.show()
    results_df = pd.DataFrame(results, columns=['Final Return', 'Max Drawdown', 'Sharpe'])
    results_df['ret_to_dd'] = results_df['Final Return'] / results_df['Max Drawdown']
    results_df['final'] = results_df['ret_to_dd'] * results_df['Sharpe']
    results_df = results_df.sort_values(by='final', ascending=False)
    #print(results_df)
    better_results = round((len(results_df[results_df['final'] > final_])/len(results_df))*100, 2)
    #print(f"Montecarlo has {better_results} % better results than base strategy.")
    if better_results < 50:
        return True
    return False


def outlier_replacement(df, column_name, percentile=0.03):
    """
    Odrzuca wartości odstające w kolumnie DataFrame i wypełnia je pozostałymi
    najmniejszą i największą wartością.
    Args:
        df (pd.DataFrame): DataFrame wejściowy.
        column_name (str): Nazwa kolumny, w której mają być zastąpione wartości.
        percentile (float): Procent wartości do odrzucenia z obu końców rozkładu.

    Returns:
        pd.DataFrame: DataFrame z zastąpionymi wartościami.
    """
    df = df.copy()  # Tworzymy kopię, aby nie modyfikować oryginalnego DataFrame
    # Obliczamy progi (kwantyle)
    lower_threshold = df[column_name].quantile(percentile)
    upper_threshold = df[column_name].quantile(1 - percentile)
    # Znajdujemy wartości *poza* progami (czyli te do odrzucenia)
    outliers_lower = df[column_name] < lower_threshold
    outliers_upper = df[column_name] > upper_threshold
    # Znajdujemy wartości *wewnątrz* progów (czyli te, które zostają)
    values_within_range = df[column_name][~(outliers_lower | outliers_upper)]
    # Określamy najmniejszą i największą wartość z tych, które zostają
    min_val = values_within_range.min()
    max_val = values_within_range.max()
    # Zastępujemy wartości odstające
    df.loc[outliers_lower, column_name] = min_val
    df.loc[outliers_upper, column_name] = max_val
    return df[column_name]



def drift_function(results_from_df):
    drift = "Neutral"
    dict_with_returns = {}
    dict_with_floats = {}
    dict_with_returns['base_res'] = [i for i in results_from_df if i!=0]
    dict_with_floats['base_sum'] = sum(results_from_df)
    br = dict_with_returns['base_res']
    dict_with_returns['courage_after_loss_only'] = [br[0]]+[float(br[i]) if br[i-1]<0 else 0 for i in range(1, len(br))]
    dict_with_floats['courage_after_loss_only_sum'] = sum(dict_with_returns['courage_after_loss_only'])
    dict_with_returns['courage_after_win_only'] = [br[0]]+[float(br[i]) if br[i-1]>0 else 0 for i in range(1, len(br))]
    dict_with_floats['courage_after_win_only_sum'] = sum(dict_with_returns['courage_after_win_only'])

    # dict_with_returns['courage_after_two_only_loss'] = br[:2]+[float(br[i])*2 if br[i-1]<0 and br[i-2]<0 else 0 for i in range(2, len(br))]
    # dict_with_floats['courage_after_two_only_loss_sum'] = sum(dict_with_returns['courage_after_two_only_loss'])
    # dict_with_returns['courage_after_two_only_win'] = br[:2]+[float(br[i])*2 if br[i-1]>0 and br[i-2]>0 else 0 for i in range(2, len(br))]
    # dict_with_floats['courage_after_two_only_win_sum'] = sum(dict_with_returns['courage_after_two_only_win'])

    # "Neutral" "Strong loss" "Weak loss" "Strong win" "Weak win"
    if dict_with_floats['courage_after_loss_only_sum'] > dict_with_floats['courage_after_win_only_sum']:
        if dict_with_floats['courage_after_loss_only_sum'] > dict_with_floats['base_sum']:
            drift = "Strong loss"
        else:
            drift = "Weak loss"
    elif dict_with_floats['courage_after_loss_only_sum'] < dict_with_floats['courage_after_win_only_sum']:
        if dict_with_floats['courage_after_win_only_sum'] > dict_with_floats['base_sum']:
            drift = "Strong win"
        else:
            drift = "Weak win"
    else:
        pass

    return drift


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
    longs = []
    shorts = []
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
            if result != 0:
                longs.append(result)
        else:
            result = (open_price - close_price) / open_price
            min_result = (max_price - open_price) / open_price
            max_result = (open_price - min_price) / open_price
            if result != 0:
                shorts.append(result)

        if result != 0:
            stats.append((result, min_result, max_result))

    mean_long = round(np.mean(longs), 8)
    mean_short = round(np.mean(shorts), 8)

    #cond = (divider then position is not ->, if this position divider == 1)
    _difference = 2
    if mean_long > 0 and mean_short < 0:
        cond = (_difference, 'long')
    elif mean_long < 0 and mean_short > 0:
        cond = (_difference, 'short')
    elif mean_long < 0 and mean_short < 0:
        cond = (1, 'both')
    elif mean_long > 0 and mean_short > 0:
        value = mean_long/mean_short
        if value > _difference:
            cond = (_difference, 'long')
        elif value < 0.5:
            cond = (_difference, 'short')
        elif value < 1:
            cond = (round(1/value, 3), 'short')
        elif value > 1:
            cond = (round(value, 3), 'long')
        else:  cond = (1, 'both')
    else: cond  = (1, 'both')

    # --- creating dataframe with statistics ---
    stats_df = pd.DataFrame(stats, columns=['result', 'min_result', 'max_result'])
    drift = drift_function(stats_df['result'].tolist())
    percent_best = 2
    number = round((percent_best/100)*len(stats_df))
    # --- normalize data to get tp and sl ---
    max_res = stats_df.sort_values(by=['max_result'], ascending=False)['max_result'][number:number]
    aberration_max = max_res.mean() + max_res.std()
    stats_df['result'] = np.where(stats_df['result'] > aberration_max, aberration_max, stats_df['result'])

    winners = stats_df[stats_df['result'] > 0]
    series_tp = outlier_replacement(winners, 'max_result', percentile=0.02)
    series_sl = outlier_replacement(winners, 'min_result', percentile=0.02)
    mean_tp = series_tp.mean()
    mean_sl = series_sl.mean()
    tp_plus_std = mean_tp - 0.1*series_tp.std()
    sl_plus_std = mean_sl + 0.1*series_sl.std()

    risk_reward_ratio = round(tp_plus_std / sl_plus_std, 3)

    try:
        win_loss_ratio = round(len(stats_df[stats_df['result'] > 0])/len(stats_df[stats_df['result'] < 0]), 3)
    except ZeroDivisionError:
        win_loss_ratio = 1
    end_result = round(risk_reward_ratio * win_loss_ratio, 2)
    if (end_result == np.inf) or (end_result > 2):
        end_result = 2.0

    garch = garch_metric(stats_df['result'])
    package = (mean_tp, mean_sl, tp_plus_std, sl_plus_std)
    package = [round(float(i), 4) for i in package] + [drift, cond]
    return round(end_result*garch*(tp_plus_std/sl_plus_std), 5), package


def vol_cond_result(cond, posType):
    divider_ = cond[0]
    protect_type = cond[1]
    if (protect_type == 'long' and posType == 0) or \
        (protect_type == 'short' and posType == 1) or \
            protect_type == 'both':
        return 1
    else:
        return divider_


def calc_result(df, sharpe_multiplier, check_week_ago=False, check_end_result=False):
    if check_week_ago:
        today = dt.now().date()
        week_ago_date = today - timedelta(days=7)
        df = df[(df['date_xy'] >= week_ago_date)]
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    cross = int(df['cross'].sum()) ** 0.85 + 2
    days = len(list(np.unique(df['date_xy'])))
    x = sharpe_multiplier/cross
    sharpe = 1
    final_result = strategy_score(df['return'], x, days)
    if check_end_result:
        end_result, risk_data = wlr_rr(df)
        return sharpe, final_result, end_result, risk_data
    else:
        return sharpe, final_result


def calc_result_metric(df, metric, check_week_ago=False, check_end_result=True):
    if check_week_ago:
        today = dt.now().date()
        week_ago_date = today - timedelta(days=7)
        df = df[(df['date_xy'] >= week_ago_date)]
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    final_result = metric(df)
    if final_result < 0:
        return 0, 0, [0 for _ in range(5)]+[(1, 'both')]
    if check_end_result:
        end_result, risk_data = wlr_rr(df)
        return final_result, end_result, risk_data
    else:
        return final_result


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


def play_with_trend(symbol, short, long, dev, divider):
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
            return 0.55 / divider
        elif d.close < d.ma_432:
            print("Good to open long!")
            return 0.35 / divider
        elif d.close > d.ma_432 and d.close < d.boll_up_432:
            print("Just long trend!")
            return 0.2 / divider
        elif d.close > d.boll_up_432:
            print("Overbougth long trend!")
            return 0.05 / divider
    else:
        if d.close > d.boll_up_432:
            print("Best to open short!")
            return -0.55 / divider
        elif d.close > d.ma_432:
            print("Good to open short!")
            return -0.35 / divider
        elif d.close < d.ma_432 and d.close > d.boll_down_432:
            print("Just short trend!")
            return -0.2 / divider
        elif d.close < d.boll_down_432:
            print("Oversold short trend!")
            return -0.05 / divider
    return 0


def play_with_trend_bt(symbol):
    #return 10, 20, 2, 1
    longs = range(1150, 1601, 25)#range(1100, 2001, 25)
    shorts = range(350, 621, 20)
    devs = range(10, 23, 2)
    df_raw = get_data(symbol, 'M5', 1, 25000)
    df_raw['weekday'] = df_raw.time.dt.weekday

    today = dt.now().weekday()
    if today > 4:
        today = 4

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


                    get_this_fuck_out = []
                    for i in range(5):
                        dfx = df[df['weekday'] == i]
                        sharpex1 = round(((dfx['return'].mean()/dfx['return'].std())), 6)
                        omegax1 = omega_ratio(dfx['return'])
                        df3 = dfx.copy()[int(len(dfx)/2):]
                        sharpex2 = round(((df3['return'].mean()/df3['return'].std())), 6)
                        omegax2 = omega_ratio(df3['return'])
                        result = np.mean([sharpex1, sharpex2])*np.mean([omegax1, omegax2])
                        get_this_fuck_out.append((i, result))

                    get_this_fuck_out = sorted(get_this_fuck_out, key=lambda x: x[1], reverse=True)
                    get_this_fuck_out = [i[0] for i in get_this_fuck_out]
                    ind_ = get_this_fuck_out.index(today)

                    match ind_:
                        case 0: divider = 1
                        case 1: divider = 1.1
                        case 2: divider = 1.2
                        case 3: divider = 1.3
                        case 4: divider = 1.4

                    results.append((dev, short, long, divider, result))
                except Exception as e:
                    print(e)
                    continue

    final = sorted(results, key=lambda x: x[4], reverse=True)[0]
    dev = final[0]
    short = final[1]
    long = final[2]
    divider = final[3]
    print(f'play_with_trend_bt results: dev={dev}, short={short}, long={long}, divider={divider}, result={round(final[4],5)}')
    return short, long, dev, divider


def get_last_closed_position_direction(symbol):
    today = dt.now().date()
    from_date = dt(today.year, today.month, today.day)
    to_date = dt.now()

    history = mt.history_deals_get(from_date, to_date)
    if history is None:
        print("Brak historii transakcji")
        return None, None

    closed_positions = sorted([deal for deal in history if deal.symbol == symbol and deal.type in (mt.DEAL_TYPE_BUY, mt.DEAL_TYPE_SELL)], key=lambda x: x.time, reverse=True)

    if not closed_positions:
        print("Brak zamkniętych pozycji dla symbolu", symbol)
        return None, None

    last_deal = closed_positions[0]
    return int(0) if last_deal.type == mt.DEAL_TYPE_BUY else int(1), last_deal.price


def volume_metrics_data():
    with open(os.path.join('volume_metrics', 'm10.json'), "r") as file:
        data = json.load(file)
    print(data)  # Otrzymasz listę list


def find_density_results(data, epsilon=10):
    """
    Dodaje do każdego punktu w `data` wartość zagęszczenia punktów w jego pobliżu,
    obliczoną na podstawie odległości euklidesowej w przestrzeni (fast, slow).

    :param data: lista list [fast, slow, result, actual_condition, daily_return, end_result, risk_data]
    :param epsilon: promień określający "bliskość" punktów
    :return: lista list [fast, slow, result * density, actual_condition, daily_return, end_result, risk_data]
    """
    results = copy.deepcopy(data)

    # Wyciągamy tylko współrzędne fast, slow do obliczeń
    positions = np.array([[row[0], row[1]] for row in data], dtype=float)

    densities = []
    for i, point in enumerate(positions):
        distances = np.linalg.norm(positions - point, axis=1)
        density = np.sum(distances < epsilon) - 1  # nie liczymy siebie
        densities.append(density)

    # Mnożenie result przez density
    for row, density in zip(results, densities):
        row[2] *= density

    return results


def strategy_rsi(df, factor, leverage, backtest=False):
    df['realrsi2'] = df.ta.rsi(length=2)
    df['realrsi2_mean'] = df['realrsi2'].rolling(factor).mean()
    df['realrsi2_std'] = df['realrsi2'].rolling(factor).std()
    df['real_boll_up'] = df['realrsi2_mean'] + df['realrsi2_std']
    df['real_boll_down'] = df['realrsi2_mean'] - df['realrsi2_std']
    df['condl'] = df['close'].rolling(factor).max()
    df['condl'] = np.where(df['condl']<=df['condl'].shift(), True, False)
    df['conds'] = df['close'].rolling(factor).min()
    df['conds'] = np.where(df['conds']>=df['conds'].shift(), True, False)
    df['stance'] = np.where(
        ((df['realrsi2'] > df['real_boll_up'])
        &(df['condl']))
        , 1, np.nan)
    df['stance'] = np.where(
        ((df['realrsi2'] < df['real_boll_down'])
        &(df['conds']))
        , -1, df['stance'])
    df['stance'] = df['stance'].ffill()
    if backtest:
        df, _ = calculate_strategy_returns(df, leverage)
        df['strategy'] = (1+df['return']).cumprod() - 1
        osm = only_strategy_metric(df, False)
        rpf = real_profit_factor_metric(df, False)
        #print("Metric result:", round(rpf*osm, 5))
        #miniplot(df, ['realrsi2', 'real_boll_up', 'real_boll_down'], ['strategy'])
        return rpf, osm
    else:
        return df['stance'].iloc[-1]


def rsi_condition(symbol, position, interval, results):
    intervals = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12', 'M15', 'M20', 'M30', 'H1']
    interval_for_data = "D1" if interval=="D1" else intervals[intervals.index(interval)+3]
    factor = int([i for i in results if i[0]==interval_for_data][0][1])
    df = get_data(symbol, interval_for_data, 1, 600)
    position_ = strategy_rsi(df, factor, 1, backtest=False)
    if (position_ == 1 and position == 0) or (position_ == -1 and position == 1):  # trend - buy best price (counter)
    #if (rsi >= 50 and rsi < 90 and position == 0) or (rsi <= 50 and rsi > 10 and position == 1):  # trend - buy with trend
        return True
    return False


def rsi_condition_for_tpsl(symbol, position, interval):
    intervals = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12', 'M15', 'M20', 'M30', 'H1']
    df = get_data(symbol, intervals[intervals.index(interval)+4], 1, 10)
    df['rsi'] = df.ta.rsi(length=2)
    rsi = df['rsi'].iloc[-1]
    if (rsi >= 90 and position == 1) or (rsi <= 10 and position == 0): # contra - best_price mode
        return True
    return False


def rsi_condition_backtest(symbol, intervals, leverage, bars):
    intervals_ = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12', 'M15', 'M20', 'M30', 'H1']
    real_intervals = [intervals_[intervals_.index(interval)+3] for interval in intervals] + ['D1']
    best_results_for_intervals = []
    for interval in real_intervals:
        df = get_data(symbol, interval, 1, bars)
        results = []
        for factor in trange(3, 34):
            rpf, osm = strategy_rsi(df, factor, leverage, True)
            results.append([factor, interval, rpf, osm])

        if all([i[3]<0 for i in results]):
            best = sorted(results, key=lambda x: x[2])[0]
            print(f"Best factor for {interval} is {best[0]} with result from rpf only {round(best[2], 4)}")
        else:
            results = [i for i in results if i[2]>1 and i[3]>0]
            best = sorted(results, key=lambda x: x[2]*x[3], reverse=True)[0]
            print(f"Best factor for {interval} is {best[0]} with result from rpf and osm {round(best[2]*best[3], 4)}")
        best_results_for_intervals.append([interval, best[0]])

    print(best_results_for_intervals)
    return best_results_for_intervals