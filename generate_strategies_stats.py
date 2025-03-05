# import sys
# import os

# sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config.parameters import symbols
from strategies_analize.global_strategies import *
from strategies_analize.complex_backtest_class import *
from strategies_analize.metrics import *


strategies = [rsidi_counter, macdd_counter, eng2m_counter, hhllx_counter, ema1b_trend, sup1n_trend, altre_trend, mo1to_trend, engulf_counter]
fast_intervals = ['M1', 'M2', 'M3', 'M5']
slow_intervals = ['M15', 'M20']
metrics = [only_strategy_metric, sharpe_metric, wlr_rr_metric, mix_rrsimple_metric, sharpe_drawdown_metric]#, complex_metric, min_final_strategy_metric]

if input('Fast or slow: ') == 'fast':
    print("FAST")
    print(len(symbols)*len(strategies)*len(fast_intervals)*(1+len(metrics)*0.10)/60)
    fast_test = Backtest_complex(symbols, fast_intervals, strategies, metrics, bars=16000)
    fast_test.full_analize()
    fast_test.output()
else:
    print("SLOW")
    print(len(symbols)*len(strategies)*len(fast_intervals)*(1+len(metrics)*0.10)/60)
    fast_test = Backtest_complex(symbols, slow_intervals, strategies, metrics, days=15, bars=9000)
    fast_test.full_analize()
    fast_test.output()