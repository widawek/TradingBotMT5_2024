import subprocess
from model_generator import generate_my_models
from datetime import datetime as dt
from time import sleep


while dt.now().hour < 9:
    print("Waiting until it is 8 o'clock.")
    sleep(60)
print(dt.now())


intervals = ['M5', 'M6', 'M10', 'M12']
symbols = ['EURUSD', 'USDCHF', 'GBPUSD']
leverage = 30
delete_old_models = True
positions_number = 3
max_reducer = 15
min_reducer = 6
volatility_divider = [sorted(list(set([i*n for i in [int(_[1:]) for _ in intervals] for n in [_ for _ in range(6, 27, 2)]]))), max_reducer, min_reducer]  
print(volatility_divider)
generate_my_models(symbols, intervals, leverage, delete_old_models)
symbols = [(symbol, 0) for symbol in symbols]
positions_values = [positions_number for _ in range(len(symbols))]  # Lista wartości symmetrical_positions dla każdego symbolu
volatility_values = [volatility_divider for _ in range(len(symbols))]  # Lista wartości daily_volatility_reduce dla każdego symbolu
directions = [i[1] for i in symbols]
symbols = [i[0] for i in symbols]

for symbol, direction, positions, volatility in zip(symbols, directions, positions_values, volatility_values):
    command = 'start cmd /k python -c "from simple_random_bot import Bot; bot = Bot(\'{}\', {}, {}, {}); bot.report()"'\
            .format(symbol, direction, positions, volatility)
    subprocess.Popen(command, shell=True)

input()