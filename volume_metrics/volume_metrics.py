from strategies_analize.global_strategies import *
from strategies_analize.metrics import *
from app.bot_functions import *
import MetaTrader5 as mt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas_ta as ta
from tqdm import trange
from tqdm import tqdm
import json
import warnings
from scipy.stats import linregress
import random
from itertools import combinations, product
from math import comb
from config.parameters import symbols, slow_range, fast_range
mt.initialize()
# Ignoruj wszystkie ostrzeżenia
warnings.filterwarnings('ignore')


def calc_result_modified(dfx):
    df = dfx.copy()
    df = df.dropna()
    #years = len(list(np.unique(df['time'].dt.date)))
    return strategy_score(df['return'], 1)


def returns_bt(df):
    leverage=6
    z = [len(str(x).split(".")[1])+1 for x in list(df["close"][:101])]
    divider = 10**round((sum(z)/len(z))-1)
    spread_mean = df.spread/divider
    spread_mean = spread_mean.mean()
    df["cross"] = np.where( ((df.stance == 1) & (df.stance.shift(1) != 1)) | \
                                    ((df.stance == -1) & (df.stance.shift(1) != -1)), 1, 0 )
    df['mkt_move'] = np.log(df.close/df.close.shift(1))
    df['return'] = (df.mkt_move * df.stance.shift(1) - (df["cross"] *(spread_mean)/df.open))*leverage
    df['return'] = np.where(df['time'].dt.date != df['time'].dt.date.shift(), 0, df['return'])
    return df


def columns_combination(all_cols, min_numb_of_symbols, max_results=50000):
    data = {}
    for col in all_cols[1:]:
        parts = col.split('_')
        symbol = parts[1]  # NZDUSD, GBPUSD, etc.
        if symbol not in data:
            data[symbol] = []
        data[symbol].append(col)  # Przechowujemy pełną nazwę kolumny

    symbols = list(data.keys())  # Lista unikalnych symboli
    estimated_total = 0
    all_possible_combinations = []

    for size in range(min_numb_of_symbols, len(symbols) + 1):
        num_symbol_combinations = comb(len(symbols), size)

        avg_strategies_per_symbol = [len(data[sym]) for sym in symbols]
        avg_comb_per_symbol_set = 1
        for i in range(size):
            avg_comb_per_symbol_set *= avg_strategies_per_symbol[i]

        estimated_combinations = num_symbol_combinations * avg_comb_per_symbol_set
        estimated_total += estimated_combinations

        # Jeśli mamy za dużo kombinacji, losowo wybieramy 50000 próbek
        if estimated_total > max_results:
            print(f"Szacowana liczba kombinacji: {estimated_total}, zwracamy losowe {max_results}")

            sampled_results = set()  # Używamy zbioru, aby uniknąć duplikatów
            while len(sampled_results) < max_results:
                symbol_comb = random.sample(symbols, size)  # Losowy wybór symboli
                strategy_choices = [random.choice(data[sym]) for sym in symbol_comb]  # Losowy wybór strategii
                sampled_results.add(tuple(strategy_choices))  # Dodajemy jako tuple (bo set nie obsługuje list)

            return [list(i) for i in sampled_results]

        # Normalne generowanie, jeśli nie przekroczyliśmy limitu
        for symbol_comb in combinations(symbols, size):
            strategy_choices = [data[sym] for sym in symbol_comb]
            for strat_comb in product(*strategy_choices):
                all_possible_combinations.append(list(strat_comb))

    print(f"Liczba wygenerowanych kombinacji: {len(all_possible_combinations)}")
    return all_possible_combinations


def returns_(df, symbol):
    leverage=6
    z = [len(str(x).split(".")[1])+1 for x in list(df["close"][:101])]
    divider = 10**round((sum(z)/len(z))-1)
    spread_mean = df.spread/divider
    spread_mean = spread_mean.mean()
    df["cross"] = np.where( ((df.stance == 1) & (df.stance.shift(1) != 1)) | \
                                    ((df.stance == -1) & (df.stance.shift(1) != -1)), 1, 0 )
    df['mkt_move'] = np.log(df.close/df.close.shift(1))
    df[f'return_{symbol}'] = (df.mkt_move * df.stance.shift(1) - (df["cross"] *(spread_mean)/df.open))*leverage
    df[f'return_{symbol}'] = np.where(df['time'].dt.date != df['time'].dt.date.shift(), 0, df[f'return_{symbol}'])
    df = df.rename(columns={'close':f'close_{symbol}'})
    return df[['time', f'return_{symbol}']]


def give_me_all_returns(strategies_to_bt, bars, interval):
    print("STRATEGIES TO BT:", strategies_to_bt)
    i = 0
    for strategy, interval, symbol, strategy_name, fast, slow, _ in tqdm(strategies_to_bt):
        print(strategy)
        df_raw = get_data(symbol, interval, 1, bars)
        dfx = strategy(df_raw.copy(), slow, fast, symbol)[0]
        dfx = returns_(dfx, symbol)
        dfx = dfx.rename(columns={f'return_{symbol}': f'return_{symbol}_{strategy_name}_{fast}_{slow}'})
        if i == 0:
            df_all = dfx.copy()
            i += 1
        else:
            df_all = pd.merge_asof(df_all, dfx, on='time')

    df_all = df_all.dropna()
    df_all.reset_index(drop=True, inplace=True)
    return df_all


class Backtest:
    def __init__(self, interval, max_fast: int=fast_range, max_slow: int=slow_range, bars: int=9000):
        self.interval = interval
        self.strategies = self.load_strategies_from_json()
        #self.bt_metric = globals()[self.strategies[0][3]]
        self.fast = max_fast
        self.slow = max_slow
        self.bars = bars
        self.strategies_to_test = []

    def load_strategies_from_json(self):
        with open("slow.json", "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        output = [i for i in loaded_data if i[2] == self.interval]
        for i in output:
            print(i)
        return output

    def backtest_strategies(self):
        for symbol, strategy_, _, bt_metric_ in tqdm(self.strategies):
            df_raw = get_data(symbol, self.interval, 1, self.bars)
            strategy = globals()[strategy_]
            bt_metric = globals()[bt_metric_]
            try:
                result = [strategy, self.interval]+self.strategy_bt(strategy, bt_metric, df_raw.copy(), symbol)
                self.strategies_to_test.append(result)
            except Exception as e:
                print(e)
                input()
                continue

    def strategy_bt(self, strategy, bt_metric, df_raw, symbol):
        results = []
        for slow in range(5, self.slow, 2):
            for fast in range(2, self.fast):
                try:
                    df = returns_bt(strategy(df_raw.copy(), slow, fast, symbol)[0])
                    if (len(df) > 0.8*self.bars) or (df['cross'].sum() > 7):
                        results.append([symbol, strategy.__name__, fast, slow, bt_metric(df)])
                    else:
                        continue
                except Exception as e:
                    continue
        return sorted(results, key=lambda x:x[-1])[-1]


class SymbolsByProfile:
    def __init__(self, symbols: list, interval: str, bars: int):
        self.interval = interval
        self.backtest = Backtest(interval)
        self.backtest.backtest_strategies()
        self.min_number_of_symbols = len(symbols) - 2
        self.bars = bars

    def all_returns_please(self):
        self.all_returns = give_me_all_returns(self.backtest.strategies_to_test, self.bars, self.interval)

    def best_results_dataframe(self):
        results = []
        self.all_returns_please()
        self.all_combinations = columns_combination(self.all_returns.columns, self.min_number_of_symbols)

        print("Combinations", len(self.all_combinations))

        with open('slow.json', "r") as file:
            data = json.load(file)
        df = pd.DataFrame(data, columns=['symbol', 'strategy', 'interval', 'metric'])
        group = df.groupby(['metric']).agg(counter=('metric', 'size'))
        final_metric_ = globals()[group.reset_index().sort_values(by='counter')['metric'].iloc[-1]]

        for i in tqdm(self.all_combinations):
            df = self.all_returns.copy()
            df['return'] = np.sum(df[i], axis=1)
            results.append((i, final_metric_(df, penalty=False))) # change metric for last best metric

        self.results_df_raw = pd.DataFrame(results, columns=['combination', 'sharpe_omega']).sort_values(by='sharpe_omega', ascending=False)
        self.results_df = self.results_df_raw[self.results_df_raw['sharpe_omega'] > 0.8*self.results_df_raw['sharpe_omega'].max()]
        self.results_df['len_'] = self.results_df['combination'].apply(lambda x: len(x))
        self.results_df.reset_index(drop=True, inplace=True)
        self.my_choice = self.results_df[self.results_df['len_'] == self.results_df['len_'].max()]['combination'].iloc[0]
        print(self.my_choice)

    def give_me_my_returns(self, plot_=False, choice=None):
        if choice is None:
            choice = self.my_choice
        self.best_df = self.all_returns.copy()
        columns_to_sum = [n for n in self.best_df.columns if ('return' in n and n in choice)]
        self.best_df['return'] = self.best_df[columns_to_sum].sum(axis=1)
        self.best_df['strategy'] = (1+self.best_df['return']).cumprod() - 1
        if plot_:
            plt.plot(self.best_df['strategy'])
            plt.title(label=f'{choice}')

    def generate_output(self):
        #strategies_dict = dict(zip([i.__name__ for i in self.strategies], self.strategies))
        strategy_names_ = ['_'.join(i.split('_')[2:-2]) for i in self.my_choice]
        symbols = [i.split('_')[1] for i in self.my_choice]
        #strategies = [strategies_dict[name] for name in strategy_names_]
        fasts = [int(i.split('_')[-2]) for i in self.my_choice]
        slows = [int(i.split('_')[-1]) for i in self.my_choice]
        intervals = [self.interval for _ in range(len(symbols))]
        self.output = list(zip(symbols, strategy_names_, fasts, slows, intervals))
        print(self.output)
        with open(f"{self.interval}.json", "w") as file:
            json.dump(self.output, file, indent=4)  # `indent=4` dla lepszej czytelności

    def all_the_work(self):
        self.best_results_dataframe()
        self.generate_output()


def m10_strategies():
    symbolz = SymbolsByProfile(symbols, "M10", 9000)
    symbolz.all_the_work()


def m15_strategies():
    symbolz = SymbolsByProfile(symbols, "M15", 9000)
    symbolz.all_the_work()


def m20_strategies():
    symbolz = SymbolsByProfile(symbols, "M20", 9000)
    symbolz.all_the_work()