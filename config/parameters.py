import sys
sys.path.append("..")
from app.functions import get_timezone_difference

# global params
intervals = ['M20']
symbols = [
        'BTCUSD',
        'DE40',
        'EURCAD',
        'XAGUSD',
        'EURJPY',
        'EURUSD',
        'GBPUSD',
        'USDCAD',
        'USDCHF',
        'USDJPY',
        'US30',
        'XAUUSD'
        ]

# model_generator_params
leverage = 20
delete_old_models = True
min_factor = 6
max_factor = 23
range_ = 1
morning_hour = 7
evening_hour = 24
probability_edge = 0.25
sharpe_limit = 2
kk_limit = 1.5
omega_limit = 1.1
n_estimators = 4000
lr_list = [0.05, 0.15, 0.30]
ts_list = [0.3]
factors = [_ for _ in range(min_factor, max_factor, range_)]
n_splits = 2
bars = 60000
change_hour = 15

# bot params
tz_diff = get_timezone_difference()
game_system = 'weighted_democracy'
max_number_of_models = 75
trigger_mode = 'on'
profit_factor = 1.5
position_size = 6       # percent of balance
kill_multiplier = 1.5   # loss of daily volatility by one position multiplier
tp_miner = 3
master_interval = intervals[0]
reverse_it_all = True   # bool
trigger_model_divider_factor = 5

def reverse_(symbol):
    if symbol in []:
        return 'reverse'
    return 'normal'