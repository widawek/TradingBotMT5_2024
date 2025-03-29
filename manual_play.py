import subprocess
from app.functions import timer, printer
from config.parameters import *
from volume_metrics.volume_metrics import m15_strategies, m20_strategies
import MetaTrader5 as mt
path_main = "C:/MT5_TICKMILL/terminal64.exe"
path_test = "C:/mt5_ic_markets/terminal64.exe"
mt.initialize(path=path_main, portable=True)

# from app.main import Bot;
# bot = Bot('EURJPY')
# bot.report()

printer("Symbols", symbols)
timer(7)

m15_strategies()
m20_strategies()

subprocess.Popen('start cmd /k python -c "from generate_strategies_stats import generate_fast; generate_fast()"', shell=True)
subprocess.Popen('start cmd /k python -c "from generate_strategies_stats import generate_slow; generate_slow()"', shell=True)

for symbol in symbols:
    command = 'start cmd /k python -c "from app.main import Bot; bot = Bot(\'{}\'); bot.report()"'.format(symbol)
    subprocess.Popen(command, shell=True)

input()