from functions import reduce_values, get_timezone_difference

def reverse_(symbol):
    if symbol in ['EURUSD', 'USDCHF', 'EURJPY', 'DE40']:
        return 'normal'
    return 'reverse'

intervals = ['M20']
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
        'EURCAD',
        'DE40',
        'EURGBP',
        'JP225',
        'BTCUSD'
        ]

leverage = 30
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
tz_diff = get_timezone_difference()
