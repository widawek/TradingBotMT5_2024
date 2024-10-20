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


def volume_calc(symbol: str, max_pos_margin: int, min_volume: int) -> None:
    leverage = mt.account_info().leverage
    symbol_info = mt.symbol_info(symbol)._asdict()
    price = mt.symbol_info_tick(symbol)._asdict()
    margin_min = round(((symbol_info["volume_min"] *
                    symbol_info["trade_contract_size"])/leverage) *
                    price["bid"], 2)
    account = mt.account_info()._asdict()
    max_pos_margin = round(account["balance"] * (max_pos_margin/100) /
                        (avg_daily_vol_(symbol) * 100))
    if "JP" not in symbol:
        volume = round((max_pos_margin / margin_min)) *\
                        symbol_info["volume_min"]
        print('Volume form: ', (max_pos_margin / margin_min))
    else:
        volume = round((max_pos_margin * 100 / margin_min)) *\
                        symbol_info["volume_min"]
        print('Volume form: ', (max_pos_margin * 100 / margin_min))
    if volume > symbol_info["volume_max"]:
        volume = float(symbol_info["volume_max"])
    print('Min volume: ', min_volume)
    print('Calculated volume: ', volume)
    if min_volume and (volume < symbol_info["volume_min"]):
        volume = symbol_info["volume_min"]
    return volume / symbol_info["volume_min"]

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

    trigger_model_divider = 9.5
    profit_factor = 1.5
    balance = mt.account_info().balance

    symbols = [
        'EURUSD',
        'GBPUSD',
        'USDCAD',
        'USDCHF',
        'USDJPY',
        'USDPLN',
        'US30',
        'XAUUSD',
        'EURJPY',
        ]

    symbols = list(set(symbols))

    pandas_options()
    symbols_list = []
    for symbol in symbols:
        volume_min = mt.symbol_info(symbol).volume_min
        margin_open, margin_close, real_spread_to_volatility = symbol_stats(symbol, volume_min, 1.5)
        symbols_list.append((symbol, margin_open, margin_close, real_spread_to_volatility))
    df = pd.DataFrame(symbols_list, columns=['symbol', 'margin_open', 'margin_close', 'real_spread_to_volatility'])
    vectorized = np.vectorize(volume_calc)
    df['volume'] = vectorized(df['symbol'], 6, True)
    df['profit_by_trigger'] = round(df['margin_close'] * df['volume'] / trigger_model_divider, 2)
    df['profit_by_trigger_%'] = round((df['profit_by_trigger'] *100 / balance), 2)
    df['loss_by_trigger'] = round(df['profit_by_trigger'] * profit_factor, 2)
    df['result'] = round(df['margin_open']*df['margin_close']*df['real_spread_to_volatility'],4)
    df = df.sort_values(by='real_spread_to_volatility')
    df.reset_index(drop=True, inplace=True)
    print(df)
    print("Mean profit %: ", round(df['profit_by_trigger_%'].mean(), 2))
    print(df['symbol'].to_list())
    number_ = df.margin_close.sum()
    print("All by min vol sum: ", round(df.margin_open.sum(), 2))
    print("All not good by bot killer: ", round(number_, 2))
    print("All good by bot tp killer: ", round(number_*2, 2))
    print(f"Propose start account balance: {round(number_*4, 2)}")


