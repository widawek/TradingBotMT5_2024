import pandas as pd
import traceback
from datetime import datetime as dt
import MetaTrader5 as mt
import hashlib


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