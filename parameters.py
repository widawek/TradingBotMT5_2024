from functions import reduce_values
from model_generator import min_factor, max_factor


intervals = ['M5', 'M6', 'M10', 'M12']
symbols = ['USDCAD', 'USDCHF', 'US30', 'EURUSD']
leverage = 30
delete_old_models = False
positions_number = 2
max_reducer = 12
min_reducer = 5
volatility_divider = [reduce_values(intervals, min_factor, max_factor), max_reducer, min_reducer]  
print(volatility_divider)
# absolute, weighted_democracy, ranked_democracy, just_democracy, invertedrank_democracy
game_system = 'weighted_democracy'
reverse_ = 'reverse' # 'normal' 'reverse' 'normal_mix'