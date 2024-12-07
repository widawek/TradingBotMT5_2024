import os
catalog = os.path.dirname(__file__)
catalog = os.path.dirname(catalog)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import pandas_ta as ta
from app.functions import add_comparison_columns
from typing import Union, Tuple
sys.path.append("..")
from app.decorators import validate_input_types


def data_operations(df, factor):
    df['adj'] = (df.high + df.low + df.close) / 3
    df['adj_higher1'] = np.where(df['adj'] > df['adj'].shift(1), 1, 0)
    df['adj_higher2'] = np.where(df['adj'] > df['adj'].shift(2), 1, 0)
    df['adj_higher3'] = np.where(df['adj'] > df['adj'].shift(3), 1, 0)
    df['adj_higher4'] = np.where(df['adj'] > df['adj'].shift(4), 1, 0)
    df['adj_lower1'] = np.where(df['adj'] < df['adj'].shift(1), 1, 0)
    df['adj_lower2'] = np.where(df['adj'] < df['adj'].shift(2), 1, 0)
    df['adj_lower3'] = np.where(df['adj'] < df['adj'].shift(3), 1, 0)
    df['adj_lower4'] = np.where(df['adj'] < df['adj'].shift(4), 1, 0)
    df['high_higher'] = np.where(df['high'] > df['high'].shift(1), 1, 0)
    df['low_lower'] = np.where(df['low'] < df['low'].shift(1), 1, 0)
    df['close_higher'] = np.where(df['close'] > df['close'].shift(1), 1, 0)
    df['volume_square'] = np.sin(np.log(df['volume']**2))
    df['high_square'] = np.sin(np.log(df['high']**2))
    df['low_square'] = np.sin(np.log(df['low']**2))
    df['close_square'] = np.sin(np.log(df['close']**2))
    df['high_log'] = np.log(df.high/df.high.shift(1))
    df['low_log'] = np.log(df.low/df.low.shift(1))
    df['close_log'] = np.log(df.close/df.close.shift(1))
    df['adj_log'] = np.log(df.adj/df.adj.shift(1))
    df['high_log2'] = np.log(df.high/df.high.shift(2))
    df['low_log2'] = np.log(df.low/df.low.shift(2))
    df['close_log2'] = np.log(df.close/df.close.shift(2))
    df['adj_log2'] = np.log(df.adj/df.adj.shift(2))
    df['volume_log'] = np.log(df.volume/df.volume.shift(1))
    df['volume_log2'] = np.log(df.volume/df.volume.shift(2))
    df['volatility_'] = (df['high'] - df['low'])/df['open']
    df['vola_vol'] = df['volume'] / df['volatility_']
    df['high_corr'] = df['close'].rolling(window=factor).corr(df['high'])
    df['low_corr'] = df['close'].rolling(window=factor).corr(df['low'])
    df['high_low_corr'] = df['high'].rolling(window=factor).corr(df['low'])
    df['logs_corr'] = df['close_log'].rolling(window=factor).corr(df['volume_log'])
    df['volume_mean'] = df['volume'].rolling(factor).mean()
    df['volume_std'] = df['volume'].rolling(factor).std()
    df['volatility_mean'] = df['volatility_'].rolling(factor).mean()
    df['volatility_std'] = df['volatility_'].rolling(factor).std()
    df['close_std'] = df['close'].rolling(factor).std()
    df['low_std'] = df['low'].rolling(factor).std()
    df['high_std'] = df['high'].rolling(factor).std()
    df['adj_std'] = df['adj'].rolling(factor).std()
    df['volume_pdiff'] = df['volume'].pct_change(periods=factor) * 100
    df['close_pdiff'] = df['close'].pct_change(periods=factor) * 100
    df['low_pdiff'] = df['low'].pct_change(periods=factor) * 100
    df['high_pdiff'] = df['high'].pct_change(periods=factor) * 100
    df['adj_pdiff'] = df['adj'].pct_change(periods=factor) * 100

    library = (
    dir(ta.trend) +
    dir(ta.momentum) +
    dir(ta.overlap) +
    dir(ta.volume) +
    dir(ta.statistics)
           )

    not_add = [
        'alma', 'ma', 'mcgd', 'kama', 'jma', 'vidya', 'hilo', 'vwap',
        'hma', 'ssf', 'wma', 'sinwma', 'linreg',
        'td_seq', 'qqe', 'inertia', 'coppock', 'cti', 'stc', 'psar', 'dpo',
        'tos_stdevall', 'mean_close', 'pos_volume', 'neg_volume', 'total_volume',
        ]

    try:
        for i in library:
            number_of_columns_before = len(df.columns)
            if not i.startswith('_') and i not in not_add:
                zxy = getattr(ta, i)
                pdw = getattr(df.ta, zxy.__name__)
                try:
                    fac = factor/3 if factor > 6 else 3
                    pdw(length=factor, slow=factor,
                        fast=int(fac), signal=int(fac),
                        k=factor, d=int(fac), append=True)
                except Exception:
                    pdw(append=True)
                number_of_columns_after = len(df.columns)
                numb_of_new = number_of_columns_after-number_of_columns_before
                df = add_comparison_columns(df, numb_of_new)
    except Exception as e:
        print(e)

    df.replace(np.inf, np.nan, inplace=True)
    df.replace(-np.inf, np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=['mean_close', 'pos_volume', 'neg_volume', 'total_volume'], axis=1, inplace=True)
    return df


def linreg_alma(df, direction, factor):
    col1 = df.ta.linreg(length=round(factor/4))
    col2 = df.ta.alma(length=round(factor))
    col2 = col2.shift(-round(factor))
    df['goal'] = np.where(col1 < col2, 1, -1)
    if direction == 'buy':
        df['goal'] = np.where(((col1 < col2) &
            (col1.shift(1) > col2.shift(1))), 'yes', 'no')
    elif direction == 'sell':
        df['goal'] = np.where(((col1 > col2) &
            (col1.shift(1) < col2.shift(1))), 'yes', 'no')
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    return df


def alma_solo(df, direction, factor):
    ma = ta.alma(df['adj'], length=int(factor))
    ma = ma.shift(-int(factor))
    col1 = df['close']
    col2 = ma
    df['goal'] = np.where(col1 < col2, 1, -1)
    if direction == 'buy':
        df['goal'] = np.where(((col1 < col2) &
                    (col1.shift(1) > col2.shift(1))), 'yes', 'no')
    elif direction == 'sell':
        df['goal'] = np.where(((col1 > col2) &
                    (col1.shift(1) < col2.shift(1))), 'yes', 'no')
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    return df


def t3_shift(df, direction, factor):
    ma = ta.t3(df['close'], length=int(factor), a=0.8)
    ma = ma.shift(-int(factor/3))
    col1 = df['close']
    col2 = ma
    df['goal'] = np.where(col1 < col2, 1, -1)
    if direction == 'buy':
        df['goal'] = np.where(((col1 < col2) &
                    (col1.shift(1) > col2.shift(1))), 'yes', 'no')
    elif direction == 'sell':
        df['goal'] = np.where(((col1 > col2) &
                    (col1.shift(1) < col2.shift(1))), 'yes', 'no')
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    return df


def macd_solo(df, direction, factor):
    fast = df.ta.sma(length=round(factor/3))
    slow = df.ta.sma(length=round(factor))
    signal = fast - slow
    macd = ta.sma(signal, length=int(factor/4))
    signal = signal.shift(-round(factor/3))
    col1 = macd
    col2 = signal
    df['goal'] = np.where(signal > macd, 1, -1)
    if direction == 'buy':
        df['goal'] = np.where(((col1 < col2) &
            (col1.shift(1) > col2.shift(1))), 'yes', 'no')
    elif direction == 'sell':
        df['goal'] = np.where(((col1 > col2) &
            (col1.shift(1) < col2.shift(1))), 'yes', 'no')
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    return df


def primal(df, direction, factor):
    factor = factor - 3
    up = (df.close < df.close.shift(-1)) & \
        (df.close < df.close.shift(-round(factor/2))) & \
        (df.close < df.close.shift(-factor))
    down = (df.close > df.close.shift(-1)) & \
        (df.close > df.close.shift(-round(factor/2))) & \
        (df.close > df.close.shift(-factor))
    df['goal'] = np.where(up, 1, np.NaN)
    df['goal'] = np.where(down, -1, df['goal'])
    df['goal'] = df['goal'].ffill()
    if direction == 'buy':
        df['goal'] = np.where(df['goal'] > df['goal'].shift(1), 'yes', 'no')
    elif direction == 'sell':
        df['goal'] = np.where(df['goal'] < df['goal'].shift(1), 'yes', 'no')
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    return df


def delete_model(path, fragment):
    for filename in os.listdir(path):
        if fragment in filename:
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Model removed: {file_path}")


@validate_input_types
def sortino_ratio(returns: Union[pd.Series, np.ndarray, list]) -> float:
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


@validate_input_types
def omega_ratio(returns: Union[pd.Series, np.ndarray, list], threshold: Union[float, int]=0) -> float:
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


@validate_input_types
def max_drawdown(returns: Union[pd.Series, np.ndarray, list]) -> float:
    compRet = (returns+1).cumprod()
    peak = compRet.expanding(min_periods=1).max()
    dd = (compRet/peak)-1
    return dd.min()


@validate_input_types
def kelly_criterion(returns: Union[pd.Series, np.ndarray, list]) -> float:
    good = [i for i in returns if i > 0]
    bad = [i for i in returns if i < 0]
    if len(returns) < 2 or len(good) < 2:
        return 0
    if len(bad) == 0:
        return 2
    p = len(good) / len(returns)
    q = len(bad) / len(returns)
    a = abs(sum(bad))
    b = sum(good)
    try:
        win = (b/q)
        loss = (a/p)
    except ZeroDivisionError:
        return 0
    kk = win - loss
    return kk