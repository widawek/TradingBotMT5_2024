import MetaTrader5 as mt
import pandas as pd
# display data on the MetaTrader 5 package
print("MetaTrader5 package author: ",mt.__author__)
print("MetaTrader5 package version: ",mt.__version__)
 
# establish connection to MetaTrader 5 terminal
if not mt.initialize():
    print("initialize() failed, error code =",mt.last_error())
    quit()
 
# get account currency
account_currency=mt.account_info().currency
print("Account currency:",account_currency)

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

def round_number_(symbol):
    return mt.symbol_info(symbol).digits


def avg_daily_vol_(symbol):
    df = get_data(symbol, "D1", 1, 30)
    df['avg_daily'] = (df.high - df.low) / df.open
    return df['avg_daily'].mean()

def avg_daily_vol_points(symbol):
    df = get_data(symbol, "D1", 1, 30)
    df['avg_daily'] = (df.high - df.low)
    return df['avg_daily'].mean()

def real_spread(symbol):
    s = mt.symbol_info(symbol)
    return s.spread / 10**s.digits


def initial_margin_for_min_vol(symbol):
    action = mt.ORDER_TYPE_BUY
    ask = mt.symbol_info_tick(symbol).ask
    lot = mt.symbol_info(symbol).volume_min
    return mt.order_calc_margin(action,symbol,lot,ask)

def symbol_stats(symbol, volume):
    volume_min = mt.symbol_info(symbol).volume_min
    margin_open = round(initial_margin_for_min_vol(symbol)*volume/volume_min, 2)
    volatility = avg_daily_vol_(symbol)*100
    volatility1 = avg_daily_vol_points(symbol)
    real_spread_to_volatility = round(real_spread(symbol)/volatility1, 4)
    margin_close = round(margin_open*volatility*1.5, 2)
    return margin_open, margin_close, real_spread_to_volatility


if __name__ == '__main__':
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'EURJPY',
            'GBPJPY', 'XTIUSD', 'XAGUSD', 'XAUUSD', 'XAGAUD',
            'JP225', 'DE40', 'USTEC', 'US30', 'BTCUSD', 'UK100']

    symbols_list = []
    for symbol in symbols:
        margin_open, margin_close, real_spread_to_volatility = symbol_stats(symbol)
        symbols_list.append((symbol, margin_open, margin_close, real_spread_to_volatility))
    df = pd.DataFrame(symbols_list, columns=['symbol', 'margin_open', 'margin_close', 'real_spread_to_volatility'])
    df['result'] = round(df['margin_open']*df['margin_close']*df['real_spread_to_volatility'],4)
    df = df.sort_values(by='result')
    df.reset_index(drop=True, inplace=True)
    print(df)