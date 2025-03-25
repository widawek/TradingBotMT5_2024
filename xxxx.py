import json
from strategies_analize.global_strategies import *

def volume_metrics_data(symbol, name_):
    def mini_data():
        with open(f'{name_}.json', "r") as file:
            data = json.load(file)
        for i in data:
            if i[0] == symbol:
                return i
        return []

    my_data = mini_data()
    if my_data == []:
        return None
    function_name = my_data[1]
    my_data[1] = globals()[function_name]
    return my_data


print(volume_metrics_data("CADJPY", 'M15'))
