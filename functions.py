import pandas as pd
import traceback
from datetime import datetime as dt
import MetaTrader5 as mt
import hashlib
import numpy as np
import pandas_ta as ta
import os
from itertools import product
import builtins
from time import sleep
import sys
import functools


def pandas_options():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)


def interval_time(time):
    h = time[0]
    t = int(time[1:])
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


def calculate_dominant(data, num_ranges=50):
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


def reduce_values(intervals, range_from, range_to):
    return sorted(
        list(set([i*n for i in [int(_[1:]) for _ in intervals]
        for n in [_ for _ in range(range_from, range_to, 2)]])))
