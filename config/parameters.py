import sys
sys.path.append("..")
from app.functions import get_timezone_difference, get_data
from extensions.investing_scrapper import Scraper
from datetime import datetime as dt

def divider():
    import MetaTrader5 as mt
    mt.initialize()
    curr = mt.account_info().currency
    df_ = get_data("USDPLN", "D1", 0, 1)
    last_o = float(df_['open'].iloc[-1])
    divider = 1 if curr == 'USD' else last_o
    return divider

if dt.now().weekday() not in [5, 6]:
    scraper = Scraper()
# global params

symbols:                                list = [
                                            'EURJPY',
                                            'EURUSD',
                                            'GBPUSD',
                                            'USDCHF',
                                            'USDJPY',
                                            'US30',
                                            'AUDUSD',
                                            'CADJPY',
                                            "AMD",
                                            "INTC",
                                            ]

leverage: int                           = 8
morning_hour: int                       = 7
evening_hour: int                       = 21# if not scraper.give_me_economics() else 20
bars: int                               = 30000
change_hour: int                        = 14
tz_diff                                 = get_timezone_difference()
game_system: str                        = 'weighted_democracy'
max_number_of_models: int               = 2
profit_factor: float                    = 1.5
position_size: float                    = round((100/len(symbols))*(0.6/divider()), 3)  # percent of balance 0.24 for tickmill europe broker leverage 1:30
kill_multiplier: float                  = 1.6  # loss of daily volatility by one position multiplier
tp_miner: int                           = 2
trigger_model_divider_factor: int       = 11
global_tracker_multiplier: float        = 0.15
profit_decrease_barrier: float          = 0.77
profit_increase_barrier: float          = 2.6
respect_overnight: bool                 = True
hardcore_hours: list                    = [] if dt.now().weekday() in [5, 6] else scraper.give_me_hardcore_hours()
fast_range: int                         = 21
slow_range: int                         = 63
spread_multiplier: int                  = 4
bot_backtest_bars: int                  = 17000
