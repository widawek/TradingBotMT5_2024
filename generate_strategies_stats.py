from config.parameters import symbols, slow_range, fast_range
from strategies_analize.global_strategies import *
from strategies_analize.complex_backtest_class import *
from strategies_analize.metrics import *


strategies = [avs1s_trend, atr11_counter, rsidi_counter, dnert_counter, macdd_counter, eng2m_counter, hybri_trend,
              hhllx_counter, mo1to_trend, engulf_counter, rsitr_trend, zsnew_trend, madif_trend, corrc_trend, orsio_counter, kdind_counter]
fast_intervals = ['M5', 'M6', 'M10']
slow_intervals = ['M15', 'M20']
metrics = [only_strategy_metric, profit_factor_metric, real_profit_factor_metric, combo_metric]

#self, symbols: list, intervals: list, strategies: list,
# metrics: list, days=10, max_fast: int=21, max_slow: int=62, bars: int=16000, excel=True

def generate_fast():
    print(len(symbols)*len(strategies)*len(fast_intervals)*(1+len(metrics)*0.09)*(17/16)/60)
    fast_test = Backtest_complex(symbols, fast_intervals, strategies, metrics, days=9, bars=17000, max_fast=fast_range, max_slow=slow_range, part_results=True)
    fast_test.full_analize()
    fast_test.output()

def generate_slow():
    print(len(symbols)*len(strategies)*len(slow_intervals)*(1+len(metrics)*0.15)/60)
    fast_test = Backtest_complex(symbols, slow_intervals, strategies, metrics, days=15, bars=9000, max_fast=fast_range, max_slow=slow_range, part_results=True)
    fast_test.full_analize()
    fast_test.output()


if __name__ == '__main__':
    if input('Fast or slow: ') == 'fast':
        print("FAST")
        print(len(symbols)*len(strategies)*len(fast_intervals)*(1+len(metrics)*0.09*(17/16))/60)
        input()
        fast_test = Backtest_complex(symbols, fast_intervals, strategies, metrics, days=9, bars=17000, max_fast=fast_range, max_slow=slow_range, part_results=True)
        fast_test.full_analize()
        fast_test.output()
    else:
        print("SLOW")
        print(len(symbols)*len(strategies)*len(slow_intervals)*(1+len(metrics)*0.15)/60)
        fast_test = Backtest_complex(symbols, slow_intervals, strategies, metrics, days=15, bars=9000, max_fast=fast_range, max_slow=slow_range, part_results=True)
        fast_test.full_analize()
        fast_test.output()