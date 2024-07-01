from simple_random_bot import Bot
import subprocess

symbols = ['EURUSD', 'GBPUSD', 'AUDUSD', 'USDCAD']
symbols = [(symbol, 0) for symbol in symbols]
positions_values = [3 for _ in range(len(symbols))]  # Lista wartości symmetrical_positions dla każdego symbolu
volatility_values = [15 for _ in range(len(symbols))]  # Lista wartości daily_volatility_reduce dla każdego symbolu

directions = [i[1] for i in symbols]
symbols = [i[0] for i in symbols]

for symbol, direction, positions, volatility in zip(symbols, directions, positions_values, volatility_values):
    command = 'start cmd /k python -c "from simple_random_bot import Bot; bot = Bot(\'{}\', {}, {}, {}); bot.report()"'\
            .format(symbol, direction, positions, volatility)
    subprocess.Popen(command, shell=True)

input()