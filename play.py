import subprocess
from app.functions import timer, printer
from config.parameters import *


printer("Symbols", symbols)
timer(6)

# from app.main import Bot;
# bot = Bot('GBPUSD')
# bot.report()

for symbol in symbols:
    command = 'start cmd /k python -c "from app.main import Bot; bot = Bot(\'{}\'); bot.report()"'.format(symbol)
    subprocess.Popen(command, shell=True)

input()
