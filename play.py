import subprocess
from functions import timer, want_to_delete_old_models
from model_generator import generate_my_models
from parameters import *

#generate_my_models(symbols, intervals, leverage, want_to_delete_old_models())
timer(7)

for symbol in symbols:
    command = 'start cmd /k python -c "from simple_random_bot import Bot; bot = Bot(\'{}\'); bot.report()"'.format(symbol)
    subprocess.Popen(command, shell=True)

input()