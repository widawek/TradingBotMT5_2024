from config.parameters import symbols
from strategies_analize.global_strategies import *
from strategies_analize.complex_backtest_class import *
from strategies_analize.metrics import *


strategies = [rsidi_counter, macdd_counter, avs1s_trend, eng2m_counter, hhllx_counter, ema1b_trend, sup1n_trend, altre_trend, mo1to_trend, engulf_counter]
fast_intervals = ['M1', 'M2', 'M3']
slow_intervals = ['M10', 'M20']
metrics = [only_strategy_metric, sharpe_metric, wlr_rr_metric, mix_rrsimple_metric]

print(len(symbols)*len(strategies)*len(fast_intervals)*(1+len(metrics)*0.10)/60)
full_test = Backtest_complex(symbols, fast_intervals, strategies, metrics, bars=16000)
full_test.full_analize()
full_test.output()
x = full_test.df_metrics
x = x.sort_values(by='result', ascending=False)
x.to_excel('strategy_results_Mhigh.xlsx')

input()