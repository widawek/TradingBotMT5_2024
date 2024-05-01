from simple_random_bot import Bot
import subprocess
from symbols import symbols

positions_values = [3 for _ in range(len(symbols))]  # Lista wartości symmetrical_positions dla każdego symbolu
volatility_values = [6 for _ in range(len(symbols))]  # Lista wartości daily_volatility_reduce dla każdego symbolu

for symbol, positions, volatility in zip(symbols, positions_values, volatility_values):
    command = 'start cmd /k python -c "from simple_random_bot import Bot; bot = Bot(\'{}\', {}, {}); bot.report()"'\
            .format(symbol, positions, volatility)
    subprocess.Popen(command, shell=True)