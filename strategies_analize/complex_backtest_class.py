import MetaTrader5 as mt
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import json
from datetime import timedelta
from strategies_analize.global_strategies import *
from strategies_analize.metrics import *
from config.parameters import symbols
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


def returns_bt_full_anal(df_raw):
    df = df_raw.copy()
    leverage=6
    z = [len(str(x).split(".")[1])+1 for x in list(df["close"][:101])]
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
    df['return'] = (df.mkt_move * df.stance.shift(1) - (df["cross"] *(spread_mean)/df.open))*leverage
    return df


class Backtest_complex:
    def __init__(self, symbols: list, intervals: list, strategies: list,
                 metrics: list, days=10, max_fast: int=21, max_slow: int=62, bars: int=16000, excel=True):
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


    def generate_main_df(self, symbol, interval):
        df_raw = get_data(symbol, interval, 1, 55000)
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


    def strategy_bt(self, symbol, strategy, df_raw):
        results = []
        for slow in range(5, self.slow, 2):
            for fast in range(2, self.fast, 2):
                try:
                    df = df_raw.copy()
                    df = returns_bt(strategy(df, slow, fast, symbol)[0])
                    if (len(df) > 0.8*self.bars) or (df['cross'].sum() > 7):
                        for i in self.metrics:
                            results.append([i.__name__, symbol, fast, slow, i(df)])
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


    def full_analize(self):
        metrics_results = []
        for interval in self.intervals:
            for symbol in self.symbols:
                self.generate_main_df(symbol, interval)
                for strategy in self.strategies:

                    try:
                        results = []
                        print(symbol, strategy.__name__,interval)

                        for df_raw, df_test in tqdm(self.full_anal):
                            best_strategies = self.strategy_bt(symbol, strategy, df_raw)
                            for metric, symbol, fast, slow, _ in best_strategies:
                                df = returns_bt_full_anal(strategy(df_test, slow, fast, symbol)[0])
                                df['strategy'] = (1+df['return']).cumprod() - 1
                                results.append((metric,
                                                round(np.mean([df['strategy'].min(),
                                                               df['strategy'].max(),
                                                               df['strategy'].iloc[-1]]), 5)))

                        results_df = pd.DataFrame(results, columns=['metric', 'final_strategy'])
                        grouped_results = results_df.groupby('metric')['final_strategy']

                        # Przetwarzanie wyników dla każdej metryki osobno
                        for metric_name, metric_results in grouped_results:
                            final_sharpe = round(metric_results.mean() / metric_results.std(), 5)
                            density = round(len(metric_results[metric_results > 0]) / len(metric_results), 2)
                            final_result = round(metric_results.sum() * 100, 2)
                            fr = (1+metric_results).cumprod() - 1
                            final_result = round(fr.iloc[-1] * 100, 2)

                            metrics_results.append((
                                symbol,
                                strategy.__name__,
                                metric_name,  # Nazwa metryki poprawnie pobrana
                                interval,
                                final_sharpe,
                                final_result,
                                density
                            ))

                    except Exception as e:
                        print("loop", e)
                        continue
                    try:
                        if self.excel:
                            self.df_metrics = pd.DataFrame(metrics_results, columns=['symbol', 'strategy', 'metric', 'interval', 'sharpe', 'result', 'density'])
                            self.df_metrics = self.df_metrics.sort_values(by='result', ascending=False)
                            try:
                                self.df_metrics.to_excel("buffer_highint.xlsx")
                            except Exception:
                                self.df_metrics.to_excel("buffer_highint2.xlsx")
                    except Exception as e:
                        print('excel', e)
                        continue

        self.df_metrics = pd.DataFrame(metrics_results, columns=['symbol', 'strategy', 'metric', 'interval', 'sharpe', 'result', 'density'])


    def output(self):
        def group_to_get_metric(df):
            dfs = df[(df['sharpe'] > 0)&(df['density']>0.5)]
            dfs = dfs.groupby(['symbol', 'metric']).agg(sharpe_mean=('sharpe', 'mean'),
                                                counter=('sharpe', 'size')).reset_index()
            if ('M1' in self.intervals) or ('M2' in self.intervals) or ('M3' in self.intervals):
                dfs = dfs[dfs['counter']>=4]
            idx = dfs.groupby('symbol')['sharpe_mean'].idxmax()
            result_ = dfs.loc[idx, ['symbol', 'metric']]
            result_ = result_.reset_index(drop=True)
            return result_

        def symbol_to_metric(symbol, df):
            try:
                return [(row.metric) for row in result_.itertuples() if row.symbol == symbol][0]
            except Exception:
                return None

        df = self.df_metrics.copy()
        result_ = group_to_get_metric(df)

        df = df[(df['sharpe'] > 0) & (df['density'] > 0.5)]
        df['best_metric'] = df['symbol'].apply(lambda x: symbol_to_metric(x, result_))
        df = df[df['metric'] == df['best_metric']]
        df = df.sort_values(by=['symbol', 'sharpe'])
        df = df[['strategy', 'symbol', 'interval', 'metric']]
        df.reset_index(drop=True, inplace=True)

        to_json = [(row.symbol, row.strategy, row.interval, row.metric) for row in df.itertuples()]

        # Zapis do pliku JSON
        name_ = 'fast' if 'M1' in self.intervals else 'slow' if 'M10' in self.intervals else 'dontknow'

        if name_ == 'fast':
            df.to_excel('fast.xlsx')
        elif name_ == 'slow':
            df.to_excel('slow.xlsx')

        with open(f"{name_}.json", "w+", encoding="utf-8") as f:
            json.dump(to_json, f, indent=4)


if __name__ == '__main__':
    pass
