import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime as dt
import MetaTrader5 as mt
import hashlib
import os
from collections import Counter
from time import sleep
import sys
from typing import Union, Tuple
sys.path.append("..")
from app.decorators import validate_input_types


def pandas_options() -> None:
    """
    Configures pandas display options for better readability.
    
    This function sets the following options:
    - display.max_columns: Displays all columns without truncation.
    - display.max_rows: Displays all rows without truncation.
    - display.max_colwidth: Displays the full width of cell values.
    """
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)


@validate_input_types
def interval_time(interval_string: str) -> int:
    """
    Converts a time interval string into its equivalent duration in minutes.
    
    The input string should start with a single-character unit indicator 
    ('M', 'H', 'D', 'W'), followed by a numeric value. For example:
    - 'M10' represents 10 minutes.
    - 'H2' represents 2 hours.
    - 'D1' represents 1 day.
    - 'W3' represents 3 weeks.

    Args:
        interval_string (str): Time interval in the format <unit><value>.

    Returns:
        int: The equivalent duration in minutes.

    Raises:
        ValueError: If the input format is invalid or the unit is unsupported.
    """
    h = interval_string[0]
    t = int(interval_string[1:])
    x = {"M": 1, "H": 60, "D": 1440, "W": 10800}
    return int(t * x[h])
    

@validate_input_types
def timeframe_(tf: str):
    """
    Retrieves the corresponding TIMEFRAME constant from the MetaTrader (mt) module.

    Args:
        tf (str): The string representation of the timeframe (e.g., 'M1', 'H1', 'D1').

    Returns:
        Any: The corresponding TIMEFRAME constant from the mt module.

    Raises:
        AttributeError: If the specified timeframe is not found in the mt module.
    """
    try:
        return getattr(mt, f'TIMEFRAME_{tf}')
    except AttributeError as e:
        raise AttributeError(f"Invalid timeframe '{tf}'. Ensure it matches a valid TIMEFRAME constant in the mt module.") from e


@validate_input_types
def get_data_for_model(symbol: str, tf: str, start: int, counter: int) -> pd.DataFrame:
    """
    Fetches historical market data for a given symbol and timeframe, 
    formats it into a DataFrame, and prepares it for model usage.

    Args:
        symbol (str): The financial instrument symbol (e.g., 'EURUSD').
        tf (str): The timeframe string (e.g., 'M1', 'H1', 'D1').
        start (int): The starting position for historical data retrieval.
        counter (int): The number of data points to retrieve.

    Returns:
        DataFrame: A pandas DataFrame containing the market data with the following columns:
            - time (datetime): Timestamp of the candle.
            - open (float): Open price.
            - high (float): High price.
            - low (float): Low price.
            - close (float): Close price.
            - volume (int): Volume of trades (integer type).
            - spread (int): Spread in points (integer type).

    Raises:
        ValueError: If the data cannot be retrieved or is empty.
    """
    # Retrieve data using MetaTrader5 API
    try:
        raw_data = mt.copy_rates_from_pos(symbol, timeframe_(tf), start, counter)
        # if raw_data is None or len(raw_data) == 0:
        #     raise ValueError(f"No data retrieved for symbol '{symbol}' with timeframe '{tf}'.")

        # Convert data to pandas DataFrame
        data = pd.DataFrame(raw_data)

        # Drop unnecessary columns and rename remaining ones
        data = data.drop(["real_volume"], axis=1)
        data.columns = ["time", "open", "high", "low", "close", "volume", "spread"]

        # Convert data types for better memory efficiency
        data['volume'] = data['volume'].astype('int32')
        data['spread'] = data['spread'].astype('int16')

        # Return the processed DataFrame
        return data

    except Exception as e:
        raise ValueError(f"Error retrieving data for symbol '{symbol}': {e}") from e


@validate_input_types
def get_data(symbol: str, tf: str, start: int, counter: int) -> pd.DataFrame:
    """
    Fetches and formats market data for a given symbol and timeframe.

    This function wraps `get_data_for_model` and converts the 'time' column
    from Unix timestamp to a human-readable datetime format.

    Args:
        symbol (str): The financial instrument symbol (e.g., 'EURUSD').
        tf (str): The timeframe string (e.g., 'M1', 'H1', 'D1').
        start (int): The starting position for historical data retrieval.
        counter (int): The number of data points to retrieve.

    Returns:
        DataFrame: A pandas DataFrame with market data, including a converted 'time' column.

    Raises:
        ValueError: If the data retrieval or processing fails.
    """
    try:
        # Get data using the helper function
        data = get_data_for_model(symbol, tf, start, counter)

        # Convert 'time' column to a datetime object
        data["time"] = pd.to_datetime(data["time"], unit="s")

        return data

    except Exception as e:
        raise ValueError(f"Error in get_data for symbol '{symbol}': {e}") from e


@validate_input_types
def magic_(symbol: str, comment: Union[str, float, int]) -> int:
    """
    Converts a string to an integer, using the SHA-256 hash function.
    Assigns a unique 6-digit magic number depending on the strategy name,
    symbol and interval.
    """
    if isinstance(comment, (float, int)):
        comment = str(comment)
    expression = symbol + comment
    hash_object = hashlib.sha256(expression.encode('utf-8'))
    hash_hex = hash_object.hexdigest()
    result = int(hash_hex, 16)
    return result // 10 ** (len(str(result)) - 6)


@validate_input_types
def round_number_(symbol: str) -> int:
    return mt.symbol_info(symbol).digits


@validate_input_types
def real_spread(symbol: str) -> float:
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


@validate_input_types
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


@validate_input_types
def most_common_value(tuples):
    licznik = Counter(tuples)
    return licznik.most_common(1)[0][0]


@validate_input_types
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


@validate_input_types
def get_returns(df: pd.DataFrame, symbol:str) -> Tuple[pd.DataFrame, float, float]:
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



def time_info(time_data, time_info):
    hours = int(time_data // 3600)
    minutes = int((time_data % 3600) // 60)
    seconds = int((time_data % 3600) % 60)
    time_info = f'{time_info} - {hours:02}:{minutes:02}:{seconds:02}'
    print(time_info)


def function_when_model_not_work(dfx, a, b):
    dfx['adj'] = (dfx['close'] + dfx['high'] + dfx['low']) / 3
    ma1 = dfx.ta.vwma(length=a)
    ma2 = ta.vwma(dfx['adj'], dfx['volume'], length=b)
    dfx['stance2'] = np.where(ma1>=ma2, 1, 0)
    dfx['stance2'] = np.where(ma1<ma2, -1, dfx['stance2'])
    return dfx['stance2']


def changer(what, value1, value2):
    return value1 if what == value2 else value2

