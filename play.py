import subprocess
from app.functions import timer, want_to_delete_old_models
from app.model_generator import generate_my_models
from config.parameters import *

generate_my_models(symbols, intervals, leverage, want_to_delete_old_models())
timer(7)

# from app.main import Bot;
# bot = Bot('BTCUSD')
# bot.report()

for symbol in symbols:
    command = 'start cmd /k python -c "from app.main import Bot; bot = Bot(\'{}\'); bot.report()"'.format(symbol)
    subprocess.Popen(command, shell=True)

input()