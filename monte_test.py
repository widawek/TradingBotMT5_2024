
from app.montecarlo import Montecarlo
from strategies_analize.global_strategies import *
from strategies_analize.metrics import *


symbol = "EURUSD"
interval = "M2"
bars = 5000
strategy = macdd_counter
metric = mix_rrsimple_metric

monte = Montecarlo(symbol, interval, strategy, metric, bars, 16, 2)
print(monte.final_p_value())