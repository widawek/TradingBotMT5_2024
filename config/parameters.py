from app.functions import reduce_values, get_timezone_difference
import sys
sys.path.append("..")

def reverse_(symbol):
    if symbol in []:
        return 'reverse'
    return 'normal'

intervals = ['M20']
symbols = [
        'BTCUSD',
        # 'EURJPY',
        # 'EURUSD',
        # 'GBPUSD',
        # 'USDCAD',
        # 'USDCHF',
        # 'USDJPY',
        # 'US30',
        # 'DE40',
        # 'XAUUSD',
        # 'EURCAD'
        ]


leverage = 20
delete_old_models = True
min_factor = 6
max_factor = 23
range_ = 1
game_system = 'weighted_democracy'
morning_hour = 7
evening_hour = 24
probability_edge = 0.25
sharpe_limit = 5
kk_limit = 1.5
omega_limit = 1.1
n_estimators = 4000
lr_list = [0.05, 0.15, 0.30]
ts_list = [0.3]
factors = [_ for _ in range(min_factor, max_factor, range_)]
n_splits = 2
bars = 60000
change_hour = 15
tz_diff = get_timezone_difference()
