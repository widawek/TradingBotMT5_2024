import MetaTrader5 as mt
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import json
from global_strategies import *
from metrics import *
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
    df = df[(df['date'] == date[-1])&(df['hour']>=6)&(df['hour']<23)]
    df["cross"] = np.where( ((df.stance == 1) & (df.stance.shift(1) != 1)) | \
                                    ((df.stance == -1) & (df.stance.shift(1) != -1)), 1, 0 )
    df['mkt_move'] = np.log(df.close/df.close.shift(1))
    df['return'] = (df.mkt_move * df.stance.shift(1) - (df["cross"] *(spread_mean)/df.open))*leverage
    return df


class Backtest_complex:
    def __init__(self, symbols: list, intervals: list, strategies: list,
                 metrics: list, max_fast: int=21, max_slow: int=62, bars: int=16000, excel=True):
        self.symbols = symbols
        self.metrics = metrics
        self.intervals = intervals
        self.strategies = strategies
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
            indexes = df_raw[df_raw['new_date'] == 1].index.tolist()[-21:-1]
            dates = df_raw[df_raw['new_date'] == 1]['date'].tolist()[-22:-1]
            self.dfs_to_backtest = [self.main_df.iloc[i-self.bars:i-1] for i in indexes]
            self.dfs_test = [self.main_df[(self.main_df['date'] >= dates[i-1])&(self.main_df['date'] <= dates[i])] for i in range(len(dates)) if i > 0]
        else:
            indexes = df_raw[df_raw['new_date'] == 1].index.tolist()[-21:-1]
            dates = df_raw[df_raw['new_date'] == 1]['date'].tolist()[-30:-1]
            self.dfs_to_backtest = [self.main_df.iloc[i-self.bars:i-1] for i in indexes]
            self.dfs_test = [self.main_df[(self.main_df['date'] >= dates[i-9])&(self.main_df['date'] <= dates[i])] for i in range(len(dates)) if i > 8]
            # print(self.dfs_to_backtest[-1].tail(5), self.dfs_test[-1].head(5), self.dfs_test[-1].tail(5))
            # input()
        assert len(self.dfs_to_backtest) == len(self.dfs_test), f'{len(self.dfs_to_backtest)} {len(self.dfs_test)}'
        self.full_anal = list(zip(self.dfs_to_backtest, self.dfs_test))


    def strategy_bt(self, symbol, strategy, df_raw):
        results = []
        for slow in range(5, self.slow, 3):
            for fast in range(2, self.fast, 2):
                try:
                    df = df_raw.copy()
                    df = returns_bt(strategy(df, slow, fast, symbol))
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
                                df = returns_bt_full_anal(strategy(df_test, slow, fast, symbol))
                                df['strategy'] = (1+df['return']).cumprod() - 1
                                results.append((metric, df['strategy'].iloc[-1]))

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
        osm = self.df_metrics.copy()
        osm['final'] = np.where((osm['result']>0) & (osm['sharpe']>0), osm['result']*osm['sharpe'], np.nan)
        osm=osm.dropna()
        osm = osm.sort_values(by='sharpe', ascending=False)
        osm.reset_index(drop=True, inplace=True)
        print(np.unique(osm['strategy']))

        interval_sharpe = osm.groupby('metric').agg(
        final_mean=('sharpe', 'sum'),
        count=('sharpe', 'size')  # Liczba pozycji w każdej grupie
            ).sort_values(by='final_mean', ascending=False)
        interval_sharpe = interval_sharpe.reset_index()
        interval_sharpe['final'] = interval_sharpe['final_mean'] * interval_sharpe['count']
        
        best_metric = interval_sharpe['metric'].iloc[0]
        osm = self.df_metrics.copy()
        osm = osm[osm['metric'] == best_metric]
        osm=osm.dropna()
        osm = osm.sort_values(by='sharpe', ascending=False)
        osm.reset_index(drop=True, inplace=True)
        print(np.unique(osm['strategy']))

        symbol_strategy_sharpe = osm.groupby(['symbol', 'strategy', 'interval', 'metric']).agg({'sharpe': 'mean'}).sort_values(by=['symbol', 'strategy', 'interval'])
        symbol_strategy_sharpe = symbol_strategy_sharpe[symbol_strategy_sharpe['sharpe'] > 0]
        symbol_strategy_sharpe = symbol_strategy_sharpe.sort_values(by='sharpe', ascending=False)
        n = int(len(symbol_strategy_sharpe) * 0.2)
        symbol_strategy_sharpe = symbol_strategy_sharpe.iloc[:1-n]
        symbol_strategy_sharpe = symbol_strategy_sharpe.reset_index()
        symbol_strategy_sharpe = symbol_strategy_sharpe.sort_values(by=['symbol', 'strategy', 'interval'])
        symbol_strategy_sharpe.reset_index(drop=True, inplace=True)
        print(len(symbol_strategy_sharpe), symbol_strategy_sharpe['sharpe'].mean())
        symbol_strategy_sharpe['metric'] = best_metric

        to_json = [(row.symbol, row.strategy, row.interval) for row in symbol_strategy_sharpe.itertuples()]

        # Zapis do pliku JSON
        name_ = 'fast' if 'M1' in self.intervals else 'slow' if 'M10' in self.intervals else 'dontknow'
        with open(f"{name_}.json", "w", encoding="utf-8") as f:
            json.dump(to_json, f, indent=4)


if __name__ == '__main__':
    pass
