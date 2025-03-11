import pandas as pd
import sys
sys.path.append("..")
import numpy as np
from tqdm import tqdm
from app.bot_functions import get_data, calculate_strategy_returns

def p_value(permutation_results):
    S_obs = permutation_results[0]  # Oryginalna statystyka
    if S_obs <= 0:
        return 1
    S_perm = permutation_results  # Wszystkie permutacje (łącznie z oryginalnym wynikiem)
    # Liczymy, ile wyników z permutacji jest >= od oryginalnego (dla testu jednostronnego)
    p = np.sum(np.abs(S_perm) >= np.abs(S_obs)) / len(S_perm)
    return p


def z_score(results):
    original_result = results[0]
    if original_result <= 0:
        return -3
    simulated_results = results[1:]
    """
    Oblicza Z-score dla oryginalnej strategii względem symulowanych wyników Monte Carlo.
    
    :param original_result: Wynik oryginalnej strategii (liczba)
    :param simulated_results: Lista wyników z symulacji Monte Carlo (lista liczb)
    :return: Z-score (float)
    """
    mean_simulated = np.mean(simulated_results)
    std_simulated = np.std(simulated_results, ddof=1)  # odchylenie standardowe próby
    
    if std_simulated == 0:
        return float('nan')  # Uniknięcie dzielenia przez zero
    
    return round((original_result - mean_simulated) / std_simulated, 4)


class Montecarlo:
    def __init__(self, symbol, interval, strategy, metric, bars, slow, fast, how_many=1000):
        self.symbol = symbol
        self.interval = interval
        self.strategy = strategy
        self.bt_metric = metric
        self.slow = slow
        self.fast = fast
        self.bars = bars
        self.how_many = how_many
        self.original_df = get_data(symbol, interval, 1, bars)
        self.permutated_dataframes = self.generate_permutated_dataframes()
        self.results = []

    def results_of_perms(self):
        self.results = [self.only_df_strategy(self.original_df)]
        for perm_df in tqdm(self.permutated_dataframes):
            self.results.append(self.only_df_strategy(perm_df))

    def only_df_strategy(self, df):
        df1, _ = self.strategy(df, self.slow, self.fast, self.symbol)
        df1, _ = calculate_strategy_returns(df1, 6)
        df1['strategy'] = (1+df1['return']).cumprod() - 1
        result = self.bt_metric(df1.copy())
        return [result, df1['strategy'].iloc[1000-self.bars:].tolist()]

    def correlation_condition(self, df_permuted, threshold=0.7):
        correlation_pearson = self.original_df['close'].corr(df_permuted['close'])
        # correlation_pearson_h = df_original['high'].corr(df_permuted['high'])
        # correlation_pearson_l = df_original['low'].corr(df_permuted['low'])
        correlations = [correlation_pearson]#, correlation_pearson_h, correlation_pearson_l]
        correlations = [True if i>threshold else False for i in correlations]
        return True if all(correlations) else False

    def generate_permutated_dataframes(self):
        permutated_dataframes = []
        progress_bar = tqdm(total=self.how_many)
        while len(permutated_dataframes) < self.how_many:
            df2 = self.random_permutation_ohlc()
            if self.correlation_condition(df2):
                permutated_dataframes.append(df2)
                progress_bar.update(1)  # Ręczna aktualizacja paska
        progress_bar.close()
        return permutated_dataframes

    def absolute_p_value(self):
        metric_results = [i[0] for i in self.results]
        strategies = [i[1][-1] for i in self.results]
        metric_p_value = p_value(metric_results)
        strategy_p_value = p_value(strategies)
        z_zcore_metric = z_score(metric_results)
        z_zcore_strategy = z_score(strategies)
        print("Z score for metric: ", z_zcore_metric)
        print("Z score for strategy: ", z_zcore_strategy)
        print("P value for metric: ", metric_p_value)
        print("P value for strategy: ", strategy_p_value)
        return round((0.001/np.mean([metric_p_value, strategy_p_value]))*z_zcore_strategy, 8)

    def final_p_value(self):
        self.results_of_perms()
        return self.absolute_p_value()

    def random_permutation_ohlc(self):
        """
        Generuje losową permutację OHLC, zachowując pierwszą i ostatnią cenę.

        :param df: Pandas DataFrame z kolumnami ['Open', 'High', 'Low', 'Close']
        :return: Pandas DataFrame z przetasowanymi wartościami OHLC
        """
        # Kopia danych
        df = self.original_df.copy()
        df_new = df.copy()
        # Oblicz zmiany cen (Close-to-Close)
        close_diff = df['close'].diff().dropna()  # Różnice między kolejnymi Close
        # Permutacja różnic
        permuted_diff = np.random.permutation(close_diff.values)
        # Odtworzenie nowej serii Close (zachowując pierwszą i ostatnią wartość)
        new_close = np.zeros_like(df['close'].values)
        new_close[0] = df['close'].iloc[0]
        for i in range(1, len(new_close)):
            new_close[i] = new_close[i-1] + (permuted_diff[i-1] if i < len(permuted_diff) else 0)
        # Permutacja High-Low w taki sposób, by zachować ich rozkład względem Close
        high_diff = df['high'] - df['close']
        low_diff = df['close'] - df['low']
        permuted_high_diff = np.random.permutation(high_diff[1:].values)
        permuted_low_diff = np.random.permutation(low_diff[1:].values)
        new_high = new_close + np.insert(permuted_high_diff, 0, high_diff.iloc[0])
        new_low = new_close - np.insert(permuted_low_diff, 0, low_diff.iloc[0])
        # Open jako średnia poprzedniego Close i aktualnego Close (lub inna strategia)
        new_open = (np.roll(new_close, 1) + new_close) / 2
        new_open[0] = df['open'].iloc[0]  # Zachowanie pierwszej wartości
        # Tworzenie nowego DataFrame
        df_new['open'] = new_open
        df_new['high'] = new_high
        df_new['low'] = new_low
        df_new['close'] = new_close
        return df_new

