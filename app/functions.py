import pandas as pd
import traceback
from datetime import datetime as dt
import MetaTrader5 as mt
import hashlib
import numpy as np
import pandas_ta as ta
import os
from collections import Counter
# from itertools import product
# import builtins
from time import sleep
import sys
sys.path.append("..")
# import functools


def pandas_options():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)


def interval_time(interval_string):
    h = interval_string[0]
    t = int(interval_string[1:])
    x = {"M": 1, "H": 60, "D": 1440, "W": 10800}
    return int(t * x[h])


def class_errors(func):
    def just_log(*args, **kwargs):
        symbol = args[0].symbol
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            time = dt.now()
            class_name = args[0].__class__.__name__
            function_name = func.__name__
            with open("class_errors.txt", "a") as log_file:
                log_file.write("Symbol {}, Time: {} Error in class {}, function {}:\n\n"
                            .format(symbol, time, class_name, function_name))
                traceback.print_exc(file=log_file)
            if isinstance(e, RecursionError):
                print("Exit")
                input()
                exit()
            raise e
    return just_log


def timeframe_(tf):
    return getattr(mt, 'TIMEFRAME_{}'.format(tf))


def get_data(symbol, tf, start, counter):
    data = pd.DataFrame(mt.copy_rates_from_pos(
                        symbol, timeframe_(tf), start, counter))
    data["time"] = pd.to_datetime(data["time"], unit="s")
    data = data.drop(["real_volume"], axis=1)
    data.columns = ["time", "open", "high", "low",
                    "close", "volume", "spread"]
    return data


def get_data_for_model(symbol, tf, start, counter):
    data = pd.DataFrame(mt.copy_rates_from_pos(
                        symbol, timeframe_(tf), start, counter))
    data = data.drop(["real_volume"], axis=1)
    data.columns = ["time", "open", "high", "low",
                    "close", "volume", "spread"]
    data['volume'] = data['volume'].astype('int32')
    data['spread'] = data['spread'].astype('int16')
    return data


def magic_(symbol, comment):
    """
    Converts a string to an integer, using the SHA-256 hash function.
    Assigns a unique 6-digit magic number depending on the strategy name,
    symbol and interval.
    """
    expression = symbol + comment
    hash_object = hashlib.sha256(expression.encode('utf-8'))
    hash_hex = hash_object.hexdigest()
    result = int(hash_hex, 16)
    return result // 10 ** (len(str(result)) - 6)


def round_number_(symbol):
    return mt.symbol_info(symbol).digits


def real_spread(symbol):
    s = mt.symbol_info(symbol)
    return s.spread / 10**s.digits


actions = {
    # Place an order for an instant deal with the specified parameters (set a market order)
    'deal': mt.TRADE_ACTION_DEAL,
    # Place an order for performing a deal at specified conditions (pending order)
    'pending': mt.TRADE_ACTION_PENDING,
    # Change open position Stop Loss and Take Profit
    'sltp': mt.TRADE_ACTION_SLTP,
    # Change parameters of the previously placed trading order
    'modify': mt.TRADE_ACTION_MODIFY,
    # Remove previously placed pending order
    'remove': mt.TRADE_ACTION_REMOVE,
    # Close a position by an opposite one
    'close': mt.TRADE_ACTION_CLOSE_BY
    }

pendings = {
    'long_stop': mt.ORDER_TYPE_BUY_STOP,
    'short_stop': mt.ORDER_TYPE_SELL_STOP,
    'long_limit': mt.ORDER_TYPE_BUY_LIMIT,
    'short_limit': mt.ORDER_TYPE_SELL_LIMIT,
    }


def sortino_ratio(returns):
    """
    Calculate the Sortino Ratio.

    Parameters:
    returns (numpy.ndarray or list): Array of investment returns.
    risk_free_rate (float): The risk-free rate.

    Returns:
    float: The Sortino Ratio.
    """
    returns = np.array(returns)
    downside_returns = returns[returns < 0]
    mean_return = np.mean(returns)
    downside_deviation = np.std(downside_returns, ddof=1)
    sortino_ratio = mean_return / downside_deviation
    return sortino_ratio


def omega_ratio(returns, threshold=0):
    """
    Calculate the Omega Ratio.

    Parameters:
    returns (numpy.ndarray or list): Array of investment returns.
    threshold (float): The threshold return. Default is 0.

    Returns:
    float: The Omega Ratio.
    """
    returns = np.array(returns)
    excess_returns = returns - threshold
    gains = excess_returns[excess_returns > 0]
    losses = -excess_returns[excess_returns < 0]
    omega_ratio = np.sum(gains) / np.sum(losses) if np.sum(losses) != 0 else np.inf
    return omega_ratio


def max_drawdown(returns):
    compRet = (returns+1).cumprod()
    peak = compRet.expanding(min_periods=1).max()
    dd = (compRet/peak)-1
    return dd.min()


def kelly_criterion(returns):
    good = [i for i in returns if i > 0]
    bad = [i for i in returns if i < 0]
    p = len(good) / len(returns)
    q = len(bad) / len(returns)
    a = abs(sum(bad))
    b = sum(good)
    try:
        win = (b/q)
    except ZeroDivisionError:
        return 0.1
    try:
        loss = (a/p)
    except ZeroDivisionError:
        return -0.1
    kk = win - loss
    if kk > 20:
        kk = 20
    if kk < -20:
        kk = -20
    return kk


def max_vol_times_price_price(df, window=30):
    # Obliczamy vol * price
    df['vol_times_price'] = df['close'] * df['volume']
    # Tworzymy kolumnę z sumami vol * price dla ostatnich window okresów
    df['sum_vol_times_price'] = df['vol_times_price'].rolling(window, min_periods=1).sum()
    # Znajdujemy indeks, gdzie suma jest największa
    max_index = df['sum_vol_times_price'].idxmax()
    # Pobieramy cenę przy tym indeksie
    max_price = df.loc[max_index, 'close']
    # Usuwamy tymczasowe kolumny
    df.drop(['vol_times_price', 'sum_vol_times_price'], axis=1, inplace=True)
    return max_price


def most_common_value(tuples):
    licznik = Counter(tuples)
    return licznik.most_common(1)[0][0]


def calculate_dominant(data, num_ranges=50):
    print("Positions: ", num_ranges)
    # Obliczanie minimalnej i maksymalnej wartości w zbiorze danych
    min_val = np.min(data)
    max_val = np.max(data)

    # Określenie szerokości każdego zakresu
    range_width = (max_val - min_val) / num_ranges

    # Tworzenie listy zakresów
    ranges = [(
        min_val + i*range_width, min_val + (i+1)*range_width,
     ((min_val + i*range_width)+(min_val + (i+1)*range_width))/2
     ) for i in range(num_ranges)]

    # Obliczanie średniej wartości w każdym zakresie
    dominant_values = []
    for r in ranges:
        in_range = (len([x for x in data if r[0] <= x < r[1]]), r[2])
        dominant_values.append(in_range)

    if len(dominant_values) > 0:
        overall_dominant = sorted(dominant_values, key=lambda x: x[0], reverse=True)[0][1]
    else:
        overall_dominant = None
    #print("Dominanta: ", overall_dominant)
    return overall_dominant


def is_this_curve_grow(curve, density):
    window_ = int(len(curve)*(density/100))
    max_ = curve.rolling(window=window_).max()
    min_ = curve.rolling(window=window_).min()
    maxes = np.where(max_ > max_.shift(1), 1, 0)
    mins = np.where(min_ < min_.shift(1), 1, 0)
    return round(np.sum(maxes)/np.sum(mins), 3)


def is_this_curve_grow2(curve, density):
    window_ = int(len(curve)*(density/100))
    sma_ = ta.sma(curve, length=window_)
    return np.sum(np.diff(sma_.dropna()))


def is_this_curve_grow3(prices, window_size):
    window_size = int(window_size)
    df = pd.DataFrame({'prices': prices})
    df['rolling_max'] = df['prices'].rolling(window=window_size).max()
    df['rolling_min'] = df['prices'].rolling(window=window_size).min()
    len_max = len(df[df['prices'] == df['rolling_max']])
    len_min = len(df[df['prices'] == df['rolling_min']])
    if len_min == 0:
        return np.inf
    return round(len_max / len_min, 2)


def get_returns(df, symbol):
    r_num = round_number_(symbol)
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    df['cross'].iloc[-1] = 1
    ret = df[df['cross'] == 1][['time', 'close', 'stance', 'strategy']]
    ret.reset_index(drop=True, inplace=True)
    ret['t_delta'] = ret['time'] - ret['time'].shift(1)
    ret = ret.replace(np.NaN, 0)
    ret.reset_index(drop=True, inplace=True)
    ret['return'] = ret['strategy'] - ret['strategy'].shift(1)
    positive = ret[ret['return'] > 0]
    positive['positive'] = positive['return'] * positive['close']
    negative = ret[ret['return'] < 0]
    negative['negative'] = abs(negative['return']) * negative['close']
    tp = positive['positive'].mean() + positive['positive'].std()
    tp = round(tp, r_num)
    sl = negative['negative'].mean() + negative['negative'].std()
    sl = round(sl, r_num)
    if sl == 0:
        sl = round(0.5*tp, r_num)
    return ret['return'].dropna(), tp, sl


def delete_model(path, fragment):
    for filename in os.listdir(path):
        if fragment in filename:
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Model removed: {file_path}")


def timer(hour_):
    while dt.now().hour < hour_:
        print(f"Waiting until it is {hour_} o'clock.")
        sleep(60)
    print(dt.now())


def reduce_values(intervals, range_from, range_to, range_):
    return sorted(
        list(set([i*n for i in [int(_[1:]) for _ in intervals]
        for n in [_ for _ in range(range_from, range_to, range_)]])))


def add_comparison_columns(df, x):
    # Pobieramy ostatnie x kolumn
    last_columns = df.iloc[:, -x:]

    # Iterujemy przez każdą kolumnę z ostatnich x kolumn
    for col in last_columns.columns:
        unique_list = sorted(list(df[col].unique()))
        if (unique_list == [0, 1] or unique_list == [-1, 1] or
            len(unique_list) == 1 or "_comp" in col or
            col in ['mean_close', 'pos_volume', 'neg_volume', 'total_volume']):
            continue
        diff = np.diff(last_columns[col], prepend=np.nan)
        # Tworzymy nową kolumnę z wartościami 1 lub 0
        df[f'{col}_comp'] = np.where(diff > 0, 1, 0)

    return df


def get_timezone_difference():
    import pytz
    timezone = pytz.timezone('Europe/Warsaw')
    df_time = mt.copy_rates_from_pos('BTCUSD', mt.TIMEFRAME_M1, 0, 1)
    time_mt5 = dt.fromtimestamp(df_time[0][0], timezone).hour
    now_ = dt.now(timezone).hour
    return now_ - time_mt5


def want_to_delete_old_models(test=False):
    answer = input('Do you want to generate models? ')
    if answer == "yes":
        pass
    else:
        return "no"

    delete_old_models = bool(int(input("0 if you don't want to delete old models or something else if yes: ")))
    print(delete_old_models)
    return delete_old_models


def printer(text, value, base_just=50):
    text = str(text)
    value = str(value)
    len_text = len(text)
    len_value = len(value)
    space = ' ' * (base_just-len_text-len_value)
    print(f'{text}{space}{value}')


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


def time_info(time_data, time_info):
    hours = int(time_data // 3600)
    minutes = int((time_data % 3600) // 60)
    seconds = int((time_data % 3600) % 60)
    time_info = f'{time_info} - {hours:02}:{minutes:02}:{seconds:02}'
    print(time_info)


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


def function_when_model_not_work(dfx, a, b):
    dfx['adj'] = (dfx['close'] + dfx['high'] + dfx['low']) / 3
    ma1 = dfx.ta.vwma(length=a)
    ma2 = ta.vwma(dfx['adj'], dfx['volume'], length=b)
    dfx['stance2'] = np.where(ma1>=ma2, 1, 0)
    dfx['stance2'] = np.where(ma1<ma2, -1, dfx['stance2'])
    return dfx['stance2']