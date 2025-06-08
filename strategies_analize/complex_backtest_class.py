import MetaTrader5 as mt
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import json
from datetime import timedelta
from strategies_analize.global_strategies import *
from strategies_analize.metrics import *
from config.parameters import symbols, leverage
from app.decorators import class_errors
mt.initialize()
warnings.filterwarnings('ignore')


def timeframe_(tf: str):
    return getattr(mt, f'TIMEFRAME_{tf}')


def get_data_for_model(symbol: str, tf: str, start: int, counter: int) -> pd.DataFrame:
    raw_data = mt.copy_rates_from_pos(symbol, timeframe_(tf), start, counter)
    data = pd.DataFrame(raw_data)
    data = data.drop(["real_volume"], axis=1)
    data.columns = ["time", "open", "high", "low", "close", "volume", "spread"]
    data['volume'] = data['volume'].astype('int32')
    data['spread'] = data['spread'].astype('int16')
    return data


def get_data(symbol: str, tf: str, start: int, counter: int) -> pd.DataFrame:
    data = get_data_for_model(symbol, tf, start, counter)
    data["time"] = pd.to_datetime(data["time"], unit="s")
    return data


def returns_bt(df):
    z = [len(str(x).split(".")[1])+1 for x in list(df["close"][-101:])]
    divider = 10**round((sum(z)/len(z))-1)
    spread_mean = df.spread/divider
    spread_mean = spread_mean.mean()
    df["cross"] = np.where( ((df.stance == 1) & (df.stance.shift(1) != 1)) | \
                                    ((df.stance == -1) & (df.stance.shift(1) != -1)), 1, 0 )
    df['mkt_move'] = np.log(df.close/df.close.shift(1))
    df['return'] = (df.mkt_move * df.stance.shift(1) - (df["cross"] *3*(spread_mean)/df.open))*leverage
    df['return'] = np.where(df['time'].dt.date != df['time'].dt.date.shift(), 0, df['return'])
    return df


def returns_bt_full_anal(df_raw, interval):
    df = df_raw.copy()
    z = [len(str(x).split(".")[1])+1 for x in list(df["close"][-101:])]
    divider = 10**round((sum(z)/len(z))-1)
    spread_mean = df.spread/divider
    spread_mean = spread_mean.mean()
    df['date'] = df['time'].dt.date
    df['hour'] = df['time'].dt.hour
    date = list(np.unique(df['date']))
    date.sort()
    df = df[(df['date'] == date[-1])]
    df["cross"] = np.where( ((df.stance == 1) & (df.stance.shift(1) != 1)) | \
                                    ((df.stance == -1) & (df.stance.shift(1) != -1)), 1, 0 )
    df['mkt_move'] = np.log(df.close/df.close.shift(1))
    df['return'] = (df.mkt_move * df.stance.shift(1) - (df["cross"] *3*(spread_mean)/df.open))*leverage
    df['return'] = np.where(df['time'].dt.date != df['time'].dt.date.shift(), 0, df['return'])
    # if interval not in ['M10', 'M12', 'M15', 'M20', 'M30']:
    #     start_index = df[(df['hour'] >= 6) & (df['cross'] == 1)].index.min()
    #     df = df.loc[start_index:]
    #     df.reset_index(drop=True, inplace=True)
    return df


class Backtest_complex:
    def __init__(self, symbols: list, intervals: list, strategies: list,
                 metrics: list, days=10, max_fast: int=21, max_slow: int=62,
                 part_results=False, bars: int=16000, excel=True):
        self.symbols = symbols
        self.metrics = metrics
        self.intervals = intervals
        self.strategies = strategies
        self.days = days
        self.fast = max_fast
        self.slow = max_slow
        self.bars = bars
        self.excel = excel
        self.strategies_to_test = []
        self.give_me_part_results(part_results)

    @class_errors
    def give_me_part_results(self, condition):
        if condition:
            try:
                name_ = 'slow_secure' if 'M15' in self.intervals else 'fast_secure'
                df_raw = pd.read_excel(f'{name_}.xlsx', index_col=None)
                df_raw = df_raw.drop(columns=['Unnamed: 0'], axis=1)
                self.part_results = df_raw.values.tolist()
            except Exception as e:
                print(e)
                self.part_results = []
        else:
            self.part_results = []

    @class_errors
    def generate_main_df(self, symbol, interval):
        bars_ = 55000 if interval in ['M1', 'M2', 'M3'] else 39999
        df_raw = get_data(symbol, interval, 1, bars_)
        self.test_df_raw = df_raw.copy()
        df_raw['date'] = df_raw['time'].dt.date
        dates = list(np.unique(df_raw['date']))
        dates.sort()
        df_raw = df_raw[(df_raw['date'] != dates[0])&(df_raw['date'] != dates[-1])]
        df_raw.reset_index(drop=True, inplace=True)
        self.main_df = df_raw.copy()
        df_raw['new_date'] = np.where(df_raw['date'] != df_raw['date'].shift(), 1, 0)
        if interval not in ['M10', 'M15', 'M20', 'M30']:
            indexes = df_raw[df_raw['new_date'] == 1].index.tolist()[-self.days-10:]
            dates = df_raw[df_raw['new_date'] == 1]['date'].tolist()[-self.days-10:]
            self.dfs_to_backtest = [self.main_df.iloc[i-self.bars:i] for i in indexes]
            self.dfs_test = [self.main_df[(self.main_df['date'] >= dates[i])&(self.main_df['date'] <= dates[i])] for i in range(len(dates))]
        else:
            indexes = df_raw[df_raw['new_date'] == 1].index.tolist()[-self.days-10:]
            dates = df_raw[df_raw['new_date'] == 1]['date'].tolist()[-self.days-19:]
            self.dfs_to_backtest = [self.main_df.iloc[i-self.bars:i] for i in indexes]
            self.dfs_test = [self.main_df[(self.main_df['date'] >= dates[i-9])&(self.main_df['date'] <= dates[i])] for i in range(len(dates)) if i >= 9]
            # print(self.dfs_to_backtest[-1].tail(5), self.dfs_test[-1].head(5), self.dfs_test[-1].tail(5))
            # input()
        assert len(self.dfs_to_backtest) == len(self.dfs_test), f'{len(self.dfs_to_backtest)} {len(self.dfs_test)}'
        # assert self.dfs_to_backtest[0]['time'].dt.date.iloc[-1] == self.dfs_test[0]['time'].dt.date.iloc[-1]-timedelta(days=1), f'bad dates'
        # assert self.dfs_to_backtest[-1]['time'].dt.date.iloc[-1] == self.dfs_test[-1]['time'].dt.date.iloc[-1]-timedelta(days=1), f'bad dates'
        print(self.dfs_to_backtest[0]['time'].dt.date.iloc[-1] == self.dfs_test[0]['time'].dt.date.iloc[-1]-timedelta(days=1))
        print(self.dfs_to_backtest[-1]['time'].dt.date.iloc[-1] == self.dfs_test[-1]['time'].dt.date.iloc[-1]-timedelta(days=1))
        self.full_anal = list(zip(self.dfs_to_backtest, self.dfs_test))[-self.days:]

    @class_errors
    def strategy_bt(self, symbol, strategy, df_raw):
        results = []
        for slow in range(5, self.slow, 2):
            for fast in range(2, self.fast, 2):
                try:
                    df = df_raw.copy()
                    df = returns_bt(strategy(df, slow, fast, symbol)[0])
                    if (len(df) > 0.8*self.bars) or (df['cross'].sum() > 7):
                        for i in self.metrics:
                            results.append([i.__name__, symbol, fast, slow, i(df, False)])
                    else:
                        continue
                except Exception as e:
                    continue
            best_results = []

        for metric in self.metrics:
            metric_results = [r for r in results if r[0] == metric.__name__]  # Filtrowanie po metryce
            if metric_results:
                best_results.append(max(metric_results, key=lambda x: x[-1]))  # Wybór najlepszego wyniku dla metryki

        return best_results

    @class_errors
    def full_analize(self):
        metrics_results = self.part_results
        print(metrics_results)
        for interval in self.intervals:
            for symbol in self.symbols:
                self.generate_main_df(symbol, interval)
                for strategy in self.strategies:
                    # 'symbol', 'strategy', 'metric', 'interval', 'sharpe', 'result', 'density'
                    if any([(i[0]==symbol and i[1] == strategy.__name__ and i[3] == interval) for i in metrics_results]):
                        continue
                    try:
                        results = []
                        print(symbol, strategy.__name__,interval)
                        for df_raw, df_test in tqdm(self.full_anal):
                            weekday = df_test['time'].dt.weekday.iloc[-1]
                            best_strategies = self.strategy_bt(symbol, strategy, df_raw)
                            for metric, symbol, fast, slow, _ in best_strategies:
                                df = returns_bt_full_anal(strategy(df_test, slow, fast, symbol)[0], interval)
                                # if interval not in ['M10', 'M12', 'M15', 'M20', 'M30']:
                                #     if len(df) < 600:
                                #         continue
                                df['strategy'] = (1+df['return']).cumprod() - 1
                                results.append((metric, round(df['strategy'].iloc[-1] ,6), weekday))

                        results_df = pd.DataFrame(results, columns=['metric', 'final_strategy', 'weekday'])
                        grouped_results = results_df.groupby('metric')

                        # Przetwarzanie wyników dla każdej metryki osobno
                        for metric_name, metric_results in grouped_results:
                            metric_values = metric_results['final_strategy']

                            final_sharpe = round(metric_values.mean() / metric_values.std(), 5)
                            density = round(len(metric_values[metric_values > 0]) / len(metric_values), 2)

                            fr = (1 + metric_values).cumprod() - 1
                            final_result = round(fr.iloc[-1] * 100, 2)

                            # Grupowanie wyników po weekday i obliczenie średnich wartości
                            weekday_avg = metric_results.groupby('weekday')['final_strategy'].mean()

                            # Uzupełnienie brakujących dni tygodnia zerami
                            weekday_avg = weekday_avg.reindex(range(5), fill_value=0)

                            # Przypisanie wyników do poszczególnych dni tygodnia
                            mon, tue, wed, thu, fri = weekday_avg.tolist()

                            metrics_results.append((
                                symbol,
                                strategy.__name__,
                                metric_name,
                                interval,
                                final_sharpe,
                                final_result,
                                density,
                                mon, tue, wed, thu, fri
                            ))

                    except Exception as e:
                        print("loop", e)
                        continue
                    try:
                        if self.excel:
                            self.df_metrics = pd.DataFrame(metrics_results, columns=['symbol', 'strategy', 'metric', 'interval', 'sharpe',
                                                                                     'result', 'density', 'mon', 'tue', 'wed', 'thu', 'fri'])
                            self.df_metrics = self.df_metrics.sort_values(by='result', ascending=False)
                            if 'M20' in self.intervals:
                                try:
                                    self.df_metrics.to_excel("bufferh.xlsx")
                                except Exception:
                                    self.df_metrics.to_excel("bufferh2.xlsx")
                            else:
                                try:
                                    self.df_metrics.to_excel("buffer.xlsx")
                                except Exception:
                                    self.df_metrics.to_excel("buffer2.xlsx")
                    except Exception as e:
                        print('excel', e)
                    continue

        self.df_metrics = pd.DataFrame(metrics_results, columns=['symbol', 'strategy', 'metric', 'interval',
                                                                 'sharpe', 'result', 'density', 'mon', 'tue', 'wed', 'thu', 'fri'])

    @class_errors
    def output(self):

        def group_to_get_metric(df):
            dfs = df.copy().dropna()
            dfs['meansum'] = dfs[['mon', 'tue', 'wed', 'thu', 'fri']].sum(axis=1)
            dfs = dfs[(dfs['sharpe'] > 0) & (dfs['density'] > 0.55) & (dfs['meansum'] > 0)]
            #dfs = dfs.rename(columns={'mon': '0', 'tue': '1', 'wed': '2', 'thu': '3', 'fri': '4'})
            dfs['mon'] = np.where(dfs['mon'] <= 0, -1, 1)
            dfs['tue'] = np.where(dfs['tue'] <= 0, -1, 1)
            dfs['wed'] = np.where(dfs['wed'] <= 0, -1, 1)
            dfs['thu'] = np.where(dfs['thu'] <= 0, -1, 1)
            dfs['fri'] = np.where(dfs['fri'] <= 0, -1, 1)

            dfs = dfs.groupby(['symbol', 'metric']).agg(
                sharpe_mean=('sharpe', 'mean'),
                counter=('sharpe', 'size')
            ).reset_index()

            intervals_ = [f'M{str(i)}' for i in range(1, 6)]
            filter_condition = any(interval in self.intervals for interval in intervals_)

            if filter_condition:
                dfs_high_counter = dfs[dfs['counter'] >= 4]
            else:
                dfs_high_counter = dfs

            # Znalezienie indeksów dla warunku counter >= 4, zabezpieczenie przed KeyError
            idx_high_counter = dfs_high_counter.groupby('symbol')['sharpe_mean'].idxmax()
            idx_fallback = dfs.groupby('symbol')['sharpe_mean'].idxmax()

            # Tworzenie słownika symbol -> najlepsza metryka
            best_metrics = {}

            for symbol in dfs['symbol'].unique():
                # Sprawdzamy, czy symbol jest w idx_high_counter i czy indeksy są poprawne
                if symbol in dfs_high_counter['symbol'].values and symbol in idx_high_counter.index:
                    index = idx_high_counter.loc[symbol]
                    if pd.notna(index) and index in dfs.index:
                        best_metrics[symbol] = dfs.loc[[index], ['symbol', 'metric']]
                    else:
                        best_metrics[symbol] = dfs.loc[[idx_fallback.loc[symbol]], ['symbol', 'metric']]
                else:
                    # Sprawdzamy, czy indeks fallback istnieje i jest poprawny
                    if symbol in idx_fallback.index:
                        index = idx_fallback.loc[symbol]
                        if pd.notna(index) and index in dfs.index:
                            best_metrics[symbol] = dfs.loc[[index], ['symbol', 'metric', 'mon', 'tue', 'wed', 'thu', 'fri']]

            # Łączenie wyników w jeden DataFrame
            if best_metrics:
                result_ = pd.concat(best_metrics.values(), ignore_index=True)
            else:
                result_ = pd.DataFrame(columns=['symbol', 'metric', 'mon', 'tue', 'wed', 'thu', 'fri'])

            return result_

        def symbol_to_metric(symbol, result_):
            try:
                return [(row.metric) for row in result_.itertuples() if row.symbol == symbol][0]
            except Exception:
                return None

        df = self.df_metrics.copy().dropna()
        result_ = group_to_get_metric(df)

        df['meansum'] = df[['mon', 'tue', 'wed', 'thu', 'fri']].sum(axis=1)
        df = df[(df['sharpe'] > 0) & (df['density'] > 0.55) & (df['meansum'] > 0)]
        df['mon'] = np.where(df['mon'] <= 0, -1, 1)
        df['tue'] = np.where(df['tue'] <= 0, -1, 1)
        df['wed'] = np.where(df['wed'] <= 0, -1, 1)
        df['thu'] = np.where(df['thu'] <= 0, -1, 1)
        df['fri'] = np.where(df['fri'] <= 0, -1, 1)
        df['best_metric'] = df['symbol'].apply(lambda x: symbol_to_metric(x, result_))
        df = df[df['metric'] == df['best_metric']]
        df = df.sort_values(by=['symbol', 'sharpe'])
        df = df[['strategy', 'symbol', 'interval', 'metric', 'mon', 'tue', 'wed', 'thu', 'fri']]
        df.reset_index(drop=True, inplace=True)

        to_json = [(row.symbol, row.strategy, row.interval, row.metric, row.mon, row.tue, row.wed, row.thu, row.fri) for row in df.itertuples()]

        # Zapis do pliku JSON
        name_ = 'fast' if 'M5' in self.intervals else 'slow' if 'M20' in self.intervals else 'dontknow'

        if name_ == 'fast':
            df.to_excel('fast.xlsx')
        elif name_ == 'slow':
            df.to_excel('slow.xlsx')

        with open(f"{name_}.json", "w+", encoding="utf-8") as f:
            json.dump(to_json, f, indent=4)


if __name__ == '__main__':
    pass
