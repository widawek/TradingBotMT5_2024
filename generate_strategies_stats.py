# import sys
# import os

# sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config.parameters import symbols, slow_range, fast_range
from strategies_analize.global_strategies import *
from strategies_analize.complex_backtest_class import *
from strategies_analize.metrics import *


strategies = [avs1s_trend, atr11_counter, rsidi_counter, dnert_counter, macdd_counter, eng2m_counter, hhllx_counter, ema1b_trend, sup1n_trend, mo1to_trend, engulf_counter]
fast_intervals = ['M1', 'M2', 'M3', 'M5']
slow_intervals = ['M15', 'M20']
metrics = [only_strategy_metric, sharpe_metric, wlr_rr_metric, mix_rrsimple_metric, sharpe_drawdown_metric]#, complex_metric min_final_strategy_metric]


#self, symbols: list, intervals: list, strategies: list,
# metrics: list, days=10, max_fast: int=21, max_slow: int=62, bars: int=16000, excel=True

if input('Fast or slow: ') == 'fast':
    print("FAST")
    print(len(symbols)*len(strategies)*len(fast_intervals)*(1+len(metrics)*0.10)/60)
    fast_test = Backtest_complex(symbols, fast_intervals, strategies, metrics, bars=16000, max_fast=fast_range, max_slow=slow_range)
    fast_test.full_analize()
    fast_test.output()
else:
    print("SLOW")
    print(len(symbols)*len(strategies)*len(fast_intervals)*(1+len(metrics)*0.10)/60)
    fast_test = Backtest_complex(symbols, slow_intervals, strategies, metrics, days=15, bars=9000, max_fast=fast_range, max_slow=slow_range)
    fast_test.full_analize()
    fast_test.output()