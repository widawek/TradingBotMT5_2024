import sys
sys.path.append("..")
from app.functions import get_timezone_difference

# global params

symbols:                                list = [
                                            'EURJPY',
                                            'EURUSD',
                                            'GBPUSD',
                                            'USDCAD',
                                            'USDCHF',
                                            'USDJPY',
                                            'US30',
                                            'XAUUSD',
                                            'BTCUSD',
                                            'XAGUSD',
                                            ]

#                                       model_generator_params
intervals: list                         = ['M20']
leverage: int                           = 20
delete_old_models: bool                 = True
min_factor: int                         = 6
max_factor: int                         = 23
range_: int                             = 1
morning_hour: int                       = 7
evening_hour: int                       = 24
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
change_hour: int                        = 15

#                                       bot params
tz_diff                                 = get_timezone_difference()
game_system: str                        = 'weighted_democracy'
max_number_of_models: int               = 75
trigger_mode: str                       = 'on'
profit_factor: float                    = 1.5
position_size: int                      = int((100/len(symbols))*0.8)  # percent of balance
kill_multiplier: float                  = 1.5  # loss of daily volatility by one position multiplier
tp_miner: int                           = 3
master_interval: str                    = intervals[0]
reverse_it_all: bool                    = True
trigger_model_divider_factor: int       = 7
base_fake_interval: str                 = 'M2'
use_moving_averages: bool               = False
start_trigger: str                      = 'moving_averages'

def reverse_(symbol):
    if symbol in []:
        return 'reverse'
    return 'normal'