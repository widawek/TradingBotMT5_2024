from functions import reduce_values
#from model_generator import min_factor, max_factor, range_

intervals = ['M20']
symbols = [
    'EURUSD', 'EURGBP',
    'USDCHF', 'GBPCHF'
    ]
leverage = 46
delete_old_models = True
positions_number = 2
max_reducer = 12
min_reducer = 5
min_factor = 7
max_factor = 23
range_ = 1
volatility_divider = [reduce_values(intervals, min_factor, max_factor, range_), max_reducer, min_reducer]
print(volatility_divider)
# absolute, weighted_democracy, ranked_democracy, just_democracy, invertedrank_democracy
game_system = 'weighted_democracy'
reverse_ = 'normal' # 'normal' 'reverse' 'normal_mix'

# model_generator
morning_hour = 8
evening_hour = 23
probability_edge = 0.25 #0.63
sharpe_limit = 5
kk_limit = 3
omega_limit = 1.2
n_estimators = 4000
lr_list = [0.05, 0.15, 0.3]#, 0.55]
ts_list = [0.3]
factors = [_ for _ in range(min_factor, max_factor, range_)]
n_splits = 2
bars = 120000
