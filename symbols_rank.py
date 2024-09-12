import MetaTrader5 as mt
import pandas as pd
from functions import *
# display data on the MetaTrader 5 package
print("MetaTrader5 package author: ",mt.__author__)
print("MetaTrader5 package version: ",mt.__version__)
mt.initialize()


def pandas_options():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)

# establish connection to MetaTrader 5 terminal
# if not mt.initialize():
#     print("initialize() failed, error code =", mt.last_error())
#     quit()

# get account currency
account_currency=mt.account_info().currency
print("Account currency:",account_currency)


def avg_daily_vol_(symbol):
    df = get_data(symbol, "D1", 1, 30)
    df['avg_daily'] = (df.high - df.low) / df.open
    return df['avg_daily'].mean()


def avg_daily_vol_points(symbol):
    df = get_data(symbol, "D1", 1, 30)
    df['avg_daily'] = (df.high - df.low)
    return df['avg_daily'].mean()


def initial_margin_for_min_vol(symbol):
    action = mt.ORDER_TYPE_BUY
    ask = mt.symbol_info_tick(symbol).ask
    lot = mt.symbol_info(symbol).volume_min
    return mt.order_calc_margin(action,symbol,lot,ask)


def symbol_stats(symbol, volume, kill_multiplier):
    volume_min = mt.symbol_info(symbol).volume_min
    margin_open = round(initial_margin_for_min_vol(symbol)*volume/volume_min, 2)
    volatility = avg_daily_vol_(symbol)*100
    volatility1 = avg_daily_vol_points(symbol)
    real_spread_to_volatility = round(real_spread(symbol)/volatility1, 4)
    margin_close = round(margin_open*volatility*kill_multiplier, 2)
    return margin_open, margin_close, real_spread_to_volatility


if __name__ == '__main__':
    symbols = [
        'JP225', 'USTEC', 'UK100', 'DE40', 'US30', 'AUDCAD', 'USDJPY',
        'AUDUSD', 'BTCUSD', 'USDCAD', 'AUDNZD', 'EURJPY', 'XAGAUD',
        'EURUSD', 'NZDCAD', 'XAGUSD', 'USDCHF', 'GBPJPY', 'GBPUSD',
        'EURGBP', 'GBPCHF', 'USDPLN', 'XAUUSD', 'XTIUSD', 'XAUAUD', 'XAUJPY'
        ]

    symbols = list(set(symbols))

    pandas_options()
    symbols_list = []
    for symbol in symbols:
        volume_min = mt.symbol_info(symbol).volume_min
        margin_open, margin_close, real_spread_to_volatility = symbol_stats(symbol, volume_min, 1.5)
        symbols_list.append((symbol, margin_open, margin_close, real_spread_to_volatility))
    df = pd.DataFrame(symbols_list, columns=['symbol', 'margin_open', 'margin_close', 'real_spread_to_volatility'])
    df['result'] = round(df['margin_open']*df['margin_close']*df['real_spread_to_volatility'],4)
    df = df.sort_values(by='real_spread_to_volatility')
    df.reset_index(drop=True, inplace=True)
    print(df)
    print(df['symbol'].to_list())
    number_ = df.margin_close.sum()
    print("All by min vol sum: ", round(df.margin_open.sum(), 2))
    print("All not good by bot killer: ", round(number_, 2))
    print("All good by bot tp killer: ", round(number_*2, 2))
    print(f"Propose start account balance: {round(number_*4, 2)}")


