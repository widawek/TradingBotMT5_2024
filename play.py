from simple_random_bot import Bot
import subprocess

# symbols = ["XAGAUD", "JP225", "ETHUSD", "USDJPY"]  # Lista symboli, które chcesz przekazać
# positions_values = [4, 3, 5, 5]  # Lista wartości symmetrical_positions dla każdego symbolu
# volatility_values = [10, 10, 10, 10]  # Lista wartości daily_volatility_reduce dla każdego symbolu

symbols = ['EURUSD']  # Lista symboli, które chcesz przekazać
positions_values = [8]  # Lista wartości symmetrical_positions dla każdego symbolu
volatility_values = [30]  # Lista wartości daily_volatility_reduce dla każdego symbolu

for symbol, positions, volatility in zip(symbols, positions_values, volatility_values):
    command = 'start cmd /k python -c "from simple_random_bot import Bot; bot = Bot(\'{}\', {}, {}); bot.report()"'\
            .format(symbol, positions, volatility)
    subprocess.Popen(command, shell=True)