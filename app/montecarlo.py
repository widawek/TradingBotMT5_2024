import pandas as pd
import sys
sys.path.append("..")
import numpy as np
from tqdm import tqdm
from app.bot_functions import get_data, calculate_strategy_returns
import scipy.stats as stats

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


def bootstrap_ci(data, alpha=0.05, n_bootstrap=10000):
    """
    Oblicza bootstrapowy przedział ufności dla podanych danych.

    :param data: Lista wyników statystyki (np. CAGR lub Max Drawdown z Monte Carlo).
    :param alpha: Poziom istotności (np. 0.05 dla 95% przedziału ufności).
    :param n_bootstrap: Liczba losowań bootstrapowych.
    :return: Dolna i górna granica przedziału ufności.
    """
    bootstrap_samples = np.random.choice(data, (n_bootstrap, len(data)), replace=True)
    stat_bootstrap = np.mean(bootstrap_samples, axis=1)  # Obliczenie średniej dla każdej próbki
    lower_bound = np.percentile(stat_bootstrap, (alpha / 2) * 100)
    upper_bound = np.percentile(stat_bootstrap, (1 - alpha / 2) * 100)
    return lower_bound, upper_bound


def lower_bound(data, confidence=0.95):
    """
    Oblicza dolne ograniczenie przedziału ufności dla średniej próbki.

    :param data: Lista lub tablica wartości numerycznych.
    :param confidence: Poziom ufności (domyślnie 0.95 dla 95%).
    :return: Wartość dolnego ograniczenia.
    """
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # Odchylenie standardowe z poprawką Bessela
    t_value = stats.t.ppf(1 - confidence, df=n-1)  # Wartość t dla poziomu ufności

    lower_bound_value = mean - (std_dev * t_value / np.sqrt(n))
    return lower_bound_value


class Montecarlo:
    def __init__(self, symbol, interval, strategy, metric, bars, slow, fast, permutated_dataframes: dict = {}, how_many=500, print_tqdm=True):
        self.print_tqdm = print_tqdm
        self.symbol = symbol
        self.interval = interval
        self.strategy = strategy
        self.bt_metric = metric
        self.slow = slow
        self.fast = fast
        self.bars = bars
        self.how_many = how_many
        self.raw_permutated_dataframes = permutated_dataframes
        self.original_df = get_data(symbol, interval, 1, bars) if self.raw_permutated_dataframes == {} else self.raw_permutated_dataframes[self.interval][0]
        self.permutated_dataframes = self.generate_permutated_dataframes() if self.raw_permutated_dataframes == {} else \
            self.raw_permutated_dataframes[self.interval][1:]
        self.results = []

    def results_of_perms(self):
        self.results = [self.only_df_strategy(self.original_df)]
        if self.print_tqdm:
            for perm_df in tqdm(self.permutated_dataframes):
                self.results.append(self.only_df_strategy(perm_df))
        else:
            for perm_df in self.permutated_dataframes:
                self.results.append(self.only_df_strategy(perm_df))

    def only_df_strategy(self, df):
        df1, _ = self.strategy(df, self.slow, self.fast, self.symbol)
        df1, _ = calculate_strategy_returns(df1, 6)
        df1['strategy'] = (1+df1['return']).cumprod() - 1
        result = self.bt_metric(df1.copy())
        return [result, df1['strategy'].iloc[1000-self.bars:].tolist()]

    def correlation_condition(self, df_permuted, threshold=0.3):
        correlation_pearson = self.original_df['close'].corr(df_permuted['close'])
        # correlation_pearson_h = df_original['high'].corr(df_permuted['high'])
        # correlation_pearson_l = df_original['low'].corr(df_permuted['low'])
        correlations = [correlation_pearson]#, correlation_pearson_h, correlation_pearson_l]
        correlations = [True if i>threshold else False for i in correlations]
        return True if all(correlations) else False

    def generate_permutated_dataframes(self):
        permutated_dataframes = []
        if self.print_tqdm:
            progress_bar = tqdm(total=self.how_many)
        while len(permutated_dataframes) < self.how_many:
            df2 = self.random_permutation_ohlc()
            if self.correlation_condition(df2):
                permutated_dataframes.append(df2)
                if self.print_tqdm:
                    progress_bar.update(1)  # Ręczna aktualizacja paska
        if self.print_tqdm:
            progress_bar.close()
        return permutated_dataframes

    def absolute_p_value(self, daily_volatility):
        metric_results = [i[0] for i in self.results]
        strategies = [i[1][-1] for i in self.results]
        metric_p_value = p_value(metric_results)
        strategy_p_value = p_value(strategies)
        z_zcore_metric = z_score(metric_results)
        z_zcore_strategy = z_score(strategies)
        bounds = bootstrap_ci(strategies, 0.1)
        bounds_met = bootstrap_ci(metric_results, 0.1)
        p_values_mean_to_score = (0.001/np.mean([metric_p_value, strategy_p_value]))
        bounds_mean = ((bounds[0]+bounds[1])/2)-(daily_volatility/4.1)
        nan_test = any(np.isnan(i) for i in [metric_p_value, strategy_p_value, z_zcore_metric, z_zcore_strategy])
        lower_bound_value = lower_bound(strategies)

        if bounds[0] > 0 and bounds[1] > 0:
            bounds_mean = 1
        if p_values_mean_to_score < 0 or bounds_mean < 0 or lower_bound_value < 0 or bounds[0] < -(daily_volatility) or strategy_p_value > 0.15 or z_zcore_strategy < 1 or nan_test:

            if self.print_tqdm:
                print("NOT OK")
                print("Lower bound value: ", round(lower_bound_value, 6))
                print("Z score for metric: ", z_zcore_metric)
                print("Z score for strategy: ", z_zcore_strategy)
                print("P value for metric: ", metric_p_value)
                print("P value for strategy: ", strategy_p_value)
                print(f"90% przedział ufności dla wyniku strategii: {bounds_met[0]*100:.2f}% - {bounds_met[1]*100:.2f}%")
                print(f"90% przedział ufności dla wyniku strategii: {bounds[0]*100:.2f}% - {bounds[1]*100:.2f}%")
            return -1

        if self.print_tqdm:
            print("OK")
            print("Lower bound value: ", round(lower_bound_value, 6))
            print("Z score for metric: ", z_zcore_metric)
            print("Z score for strategy: ", z_zcore_strategy)
            print("P value for metric: ", metric_p_value)
            print("P value for strategy: ", strategy_p_value)
            print(f"90% przedział ufności dla wyniku metryki: {bounds_met[0]*100:.2f}% - {bounds_met[1]*100:.2f}%")
            print(f"90% przedział ufności dla wyniku strategii: {bounds[0]*100:.2f}% - {bounds[1]*100:.2f}%")
        return round(p_values_mean_to_score*z_zcore_strategy, 8)

    def final_p_value(self, daily_volatility):
        self.results_of_perms()
        return self.absolute_p_value(daily_volatility)

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


class PermutatedDataFrames:
    def __init__(self, symbol: str, intervals: list, bars, how_many=500):
        self.symbol = symbol
        self.intervals = intervals
        self.bars = bars
        self.how_many = how_many

    def correlation_condition(self, df_raw, df_permuted, threshold=0.3):
        correlation_pearson = df_raw['close'].corr(df_permuted['close'])
        correlations = [correlation_pearson]
        correlations = [True if i>threshold else False for i in correlations]
        return True if all(correlations) else False

    def generate_permutated_dataframes(self, df_raw):
        permutated_dataframes = []
        progress_bar = tqdm(total=self.how_many)
        while len(permutated_dataframes) < self.how_many:
            df2 = self.random_permutation_ohlc(df_raw)
            if self.correlation_condition(df_raw, df2):
                permutated_dataframes.append(df2)
                progress_bar.update(1)  # Ręczna aktualizacja paska
        progress_bar.close()
        return permutated_dataframes

    def random_permutation_ohlc(self, df_raw):
        df = df_raw.copy()
        df_new = df.copy()
        close_diff = df['close'].diff().dropna()
        permuted_diff = np.random.permutation(close_diff.values)
        new_close = np.zeros_like(df['close'].values)
        new_close[0] = df['close'].iloc[0]
        for i in range(1, len(new_close)):
            new_close[i] = new_close[i-1] + (permuted_diff[i-1] if i < len(permuted_diff) else 0)
        high_diff = df['high'] - df['close']
        low_diff = df['close'] - df['low']
        permuted_high_diff = np.random.permutation(high_diff[1:].values)
        permuted_low_diff = np.random.permutation(low_diff[1:].values)
        new_high = new_close + np.insert(permuted_high_diff, 0, high_diff.iloc[0])
        new_low = new_close - np.insert(permuted_low_diff, 0, low_diff.iloc[0])
        new_open = (np.roll(new_close, 1) + new_close) / 2
        new_open[0] = df['open'].iloc[0]
        df_new['open'] = new_open
        df_new['high'] = new_high
        df_new['low'] = new_low
        df_new['close'] = new_close
        return df_new

    def dataframes_output(self):
        dataframes_for_intervals = {}
        for interval in self.intervals:
            df_raw = get_data(self.symbol, interval, 1, self.bars)
            result = [df_raw] + self.generate_permutated_dataframes(df_raw)
            dataframes_for_intervals[interval] = result
        return dataframes_for_intervals