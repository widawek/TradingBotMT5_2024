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
                                            #'USDCAD',
                                            'USDCHF',
                                            'USDJPY',
                                            'US30',
                                            'AUDUSD',
                                            'CADJPY',
                                            "AMD",
                                            "INTC",
                                            "DE40"
                                            #'AUDJPY',
                                            #'XTIUSD',
                                            ]


#                                       model_generator_params
intervals: list                         = ['M10']
leverage: int                           = 8
delete_old_models: bool                 = True
min_factor: int                         = 6
max_factor: int                         = 23
range_: int                             = 1
morning_hour: int                       = 7
evening_hour: int                       = 21# if not scraper.give_me_economics() else 20
probability_edge: float                 = 0.25
sharpe_limit: float                     = 3.0
kk_limit: float                         = 1.0
omega_limit: float                      = 1.1
n_estimators: int                       = 4000
lr_list: list                           = [0.05, 0.15, 0.30]
ts_list: list                           = [0.3]
factors: list                           = [_ for _ in range(min_factor, max_factor, range_)]
n_splits: int                           = 2
bars: int                               = 30000
change_hour: int                        = 14

#                                       bot params
tz_diff                                 = get_timezone_difference()
game_system: str                        = 'weighted_democracy'
max_number_of_models: int               = 2
profit_factor: float                    = 1.5
position_size: float                    = round((100/len(symbols))*(0.5/divider()), 3)  # percent of balance 0.24 for tickmill europe broker leverage 1:30
kill_multiplier: float                  = 1.6  # loss of daily volatility by one position multiplier
tp_miner: int                           = 2
master_interval: str                    = intervals[0]
trigger_model_divider_factor: int       = 11
base_fake_interval: str                 = 'M2'
use_moving_averages: bool               = False
start_trigger: str                      = 'moving_averages'
global_tracker_multiplier: float        = 0.15
profit_decrease_barrier: float          = 0.77
profit_increase_barrier: float          = 2.6
respect_overnight: bool                 = True
hardcore_hours: list                    = [] if dt.now().weekday() in [5, 6] else scraper.give_me_hardcore_hours()
fast_range: int                         = 21
slow_range: int                         = 63
