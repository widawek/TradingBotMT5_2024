import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    f1_score, precision_score, recall_score
    )
import MetaTrader5 as mt
import pandas_ta as ta
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from itertools import product
import warnings
warnings.filterwarnings("ignore")
mt.initialize()
import os
catalog = os.path.dirname(__file__)
catalog = os.path.dirname(catalog)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.functions import *
from config.parameters import *


def data_operations(df, factor):
    df['adj'] = (df.high + df.low + df.close) / 3
    df['adj_higher1'] = np.where(df['adj'] > df['adj'].shift(1), 1, 0)
    df['adj_higher2'] = np.where(df['adj'] > df['adj'].shift(2), 1, 0)
    df['adj_higher3'] = np.where(df['adj'] > df['adj'].shift(3), 1, 0)
    df['adj_higher4'] = np.where(df['adj'] > df['adj'].shift(4), 1, 0)
    df['adj_lower1'] = np.where(df['adj'] < df['adj'].shift(1), 1, 0)
    df['adj_lower2'] = np.where(df['adj'] < df['adj'].shift(2), 1, 0)
    df['adj_lower3'] = np.where(df['adj'] < df['adj'].shift(3), 1, 0)
    df['adj_lower4'] = np.where(df['adj'] < df['adj'].shift(4), 1, 0)
    df['high_higher'] = np.where(df['high'] > df['high'].shift(1), 1, 0)
    df['low_lower'] = np.where(df['low'] < df['low'].shift(1), 1, 0)
    df['close_higher'] = np.where(df['close'] > df['close'].shift(1), 1, 0)
    df['volume_square'] = np.sin(np.log(df['volume']**2))
    df['high_square'] = np.sin(np.log(df['high']**2))
    df['low_square'] = np.sin(np.log(df['low']**2))
    df['close_square'] = np.sin(np.log(df['close']**2))
    df['high_log'] = np.log(df.high/df.high.shift(1))
    df['low_log'] = np.log(df.low/df.low.shift(1))
    df['close_log'] = np.log(df.close/df.close.shift(1))
    df['adj_log'] = np.log(df.adj/df.adj.shift(1))
    df['high_log2'] = np.log(df.high/df.high.shift(2))
    df['low_log2'] = np.log(df.low/df.low.shift(2))
    df['close_log2'] = np.log(df.close/df.close.shift(2))
    df['adj_log2'] = np.log(df.adj/df.adj.shift(2))
    df['volume_log'] = np.log(df.volume/df.volume.shift(1))
    df['volume_log2'] = np.log(df.volume/df.volume.shift(2))
    df['volatility_'] = (df['high'] - df['low'])/df['open']
    df['vola_vol'] = df['volume'] / df['volatility_']
    df['high_corr'] = df['close'].rolling(window=factor).corr(df['high'])
    df['low_corr'] = df['close'].rolling(window=factor).corr(df['low'])
    df['high_low_corr'] = df['high'].rolling(window=factor).corr(df['low'])
    df['logs_corr'] = df['close_log'].rolling(window=factor).corr(df['volume_log'])
    df['volume_mean'] = df['volume'].rolling(factor).mean()
    df['volume_std'] = df['volume'].rolling(factor).std()
    df['volatility_mean'] = df['volatility_'].rolling(factor).mean()
    df['volatility_std'] = df['volatility_'].rolling(factor).std()
    df['close_std'] = df['close'].rolling(factor).std()
    df['low_std'] = df['low'].rolling(factor).std()
    df['high_std'] = df['high'].rolling(factor).std()
    df['adj_std'] = df['adj'].rolling(factor).std()
    df['volume_pdiff'] = df['volume'].pct_change(periods=factor) * 100
    df['close_pdiff'] = df['close'].pct_change(periods=factor) * 100
    df['low_pdiff'] = df['low'].pct_change(periods=factor) * 100
    df['high_pdiff'] = df['high'].pct_change(periods=factor) * 100
    df['adj_pdiff'] = df['adj'].pct_change(periods=factor) * 100

    library = (
    dir(ta.trend) +
    dir(ta.momentum) +
    dir(ta.overlap) +
    dir(ta.volume) +
    dir(ta.statistics)
           )

    not_add = [
        'alma', 'ma', 'mcgd', 'kama', 'jma', 'vidya', 'hilo', 'vwap',
        # 'ichimoku',
        #'supertrend',
        #'squeeze_pro'
        'hma', 'ssf', 'wma', 'sinwma', 'linreg',
        'td_seq', 'qqe', 'inertia', 'coppock', 'cti', 'stc', 'psar', 'dpo',
        'tos_stdevall', 'mean_close', 'pos_volume', 'neg_volume', 'total_volume',
        ]

    try:
        for i in library:
            number_of_columns_before = len(df.columns)
            if not i.startswith('_') and i not in not_add:
                zxy = getattr(ta, i)
                pdw = getattr(df.ta, zxy.__name__)
                try:
                    fac = factor/3 if factor > 6 else 3
                    pdw(length=factor, slow=factor,
                        fast=int(fac), signal=int(fac),
                        k=factor, d=int(fac), append=True)
                except Exception:
                    pdw(append=True)
                number_of_columns_after = len(df.columns)
                numb_of_new = number_of_columns_after-number_of_columns_before
                df = add_comparison_columns(df, numb_of_new)
    except Exception as e:
        print(e)

    df.replace(np.inf, np.nan, inplace=True)
    df.replace(-np.inf, np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    df.drop(columns=['mean_close', 'pos_volume', 'neg_volume', 'total_volume'], axis=1, inplace=True)
    return df


def stats_from_positions_returns(df, symbol, sharpe_multiplier, print_, leverage):
    status = "NO"
    annotation = "clean position"
    df['cross'] = np.where(df['stance'] != df['stance'].shift(1), 1, 0)
    density = round((df['cross'].sum()/len(df))*100, 2)
    returns, _, _ = get_returns(df, symbol)
    strategy_result = (1 + returns).cumprod() - 1

    returns_list = returns.to_list()
    sharpe = sharpe_multiplier * returns.mean()/returns.std()
    sorotino = sharpe_multiplier * sortino_ratio(returns_list)
    omega = omega_ratio(returns_list)
    dom_ret = calculate_dominant(returns_list, num_ranges=len(returns_list))
    mean_return = sharpe_multiplier * np.mean(returns_list)
    drawdown = max_drawdown(returns)
    kk = kelly_criterion(returns)
    result = round(sharpe * omega * 100, 2)

    if print_:
        print()
        print(f"Signals density {annotation}:      ", density)
        print(f"Final result {annotation}:         ", round(df['strategy'].mean() +
                                             df['strategy'].iloc[-1], 2))
        print(f"Sharpe ratio {annotation}:         ", round(sharpe, 2))
        print(f"Sorotino ratio {annotation}:       ", round(sorotino, 2))
        print(f"Omega ratio {annotation}:          ", round(omega, 4))
        print(f"Kelly ratio {annotation}:          ", round(kk, 4))
        print(f"Max drawdown {annotation}:         ", round(drawdown, 4))
        print(f"Dominant return [%] {annotation}:  ", round(dom_ret*100, 5))
        print(f"Mean return [%] {annotation}:      ", round(mean_return, 2))
        print(f"Median return [%] {annotation}:    ", round(
            sharpe_multiplier * np.median(df['return'].dropna().to_list()), 2))

    if (omega > omega_limit and sharpe > sharpe_limit and kk > kk_limit and dom_ret > 0):
        status = "YES"
        print(f"OK {annotation} "*30)
        print(f"""#### RESULT: {result} ####""")
    return status, strategy_result, drawdown


def interval_time_sharpe(interval):
    match interval[0]:
        case 'D': return np.sqrt(252)
        case 'H': return np.sqrt(252 * 24 / int(interval[1:]))
        case 'M': return np.sqrt(252 * 24 * 60 / int(interval[1:]))

def train_dataset(df, direction, parameters, factor, n_estimators, function, t_set, show_results=True, n_splits=2):
    dataset = df.copy()
    dataset = function(dataset, direction, factor)

    # Isolate the x and y variables
    y = dataset.iloc[:, -1].values
    X = dataset._get_numeric_data()

    # Isolate the categorical variables
    dataset_categorical = dataset.select_dtypes(exclude="number")

    # Transform categorical variables into dummy variables
    dataset_categorical = pd.get_dummies(data=dataset_categorical, drop_first=True)

    # Joining numerical and categorical datasets
    final_dataset = pd.concat([X, dataset_categorical], axis=1)
    feature_columns = list(final_dataset.columns.values)
    feature_columns = feature_columns[:-1]


    # Isolate the x and y variables part 2
    y = final_dataset.iloc[:, -1].values
    X = final_dataset.iloc[:, :-1].values

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Metrics to store performance across folds
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    models = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Calculate goal density (optional)
        goal_density = round(np.mean(y_test) * 100, 2)
        print(f"Goal: {goal_density} % yes/no")

        # Create xgboost matrices
        Train = xgb.DMatrix(X_train, label=y_train, feature_names=feature_columns)
        Test = xgb.DMatrix(X_test, label=y_test, feature_names=feature_columns)

        # Train the model
        model = xgb.train(
            params=parameters,
            dtrain=Train,
            num_boost_round=n_estimators,
            evals=[(Test, "Yes")],
            early_stopping_rounds=33,
            verbose_eval=False
        )
        models.append(model)
        if show_results:
            # Predictions
            predictions2 = model.predict(Test)
            predictions2 = np.where(predictions2 > probability_edge, 1, 0)

            # Confusion matrix
            confusion_matrix2 = confusion_matrix(y_test, predictions2)
            print(confusion_matrix2)
            report2 = classification_report(y_test, predictions2)
            print(report2)

            # Calculate performance metrics
            accuracy = accuracy_score(y_test, predictions2)
            precision = precision_score(y_test, predictions2)
            recall = recall_score(y_test, predictions2)
            f1 = f1_score(y_test, predictions2)

            # Append metrics to the lists
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

    # Calculate average metrics across all folds
    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)

    if show_results:
        print(f'Average Accuracy: {round(avg_accuracy, 3)}')
        print(f'Average Precision: {round(avg_precision, 3)}')
        print(f'Average Recall: {round(avg_recall, 3)}')
        print(f'Average F1 Score: {round(avg_f1, 3)}')


    return models, goal_density


def strategy_with_chart_(d_buy, d_sell, df, leverage, interval, symbol, factor,
                         chart=True, print_=True):
    status = "NO"
    sharpe_multiplier = interval_time_sharpe(interval)
    z = [len(str(x).split(".")[1])+1 for x in list(df["close"][:101])]
    divider = 10**round((sum(z)/len(z))-1)
    spread_mean = df.spread/divider
    spread_mean = spread_mean.mean()
    df['cross'] = np.where(df['stance'] != df['stance'].shift(1), 1, 0)
    density = round((df['cross'].sum()/len(df))*100, 2)
    #print(df['cross'].sum()/len(df))
    df['mkt_move'] = np.log(df.close/df.close.shift(1))
    df['return'] = np.where(df['time2'].dt.date == df['time2'].dt.date.shift(1),
                            (df.mkt_move * df.stance.shift(1) -\
                                (df.cross *(df.spread/divider)/df.open))*leverage, 0)
    df['strategy'] = (1+df['return']).cumprod() - 1
    df['max_price'] = df.apply(lambda row: max_vol_times_price_price(df.loc[:row.name]), axis=1)
    dominant = calculate_dominant(df['close'].to_list(), num_ranges=len(df['return']))
    just_line = np.linspace(df['strategy'].dropna().iloc[0], df['strategy'].dropna().iloc[-1], num=len(df['strategy'].dropna()))
    df.loc[df.index[-1] - len(df.strategy.dropna()) + 1:df.index[-1] , 'just_line'] = just_line

    df['sqrt_error'] = np.sqrt((df['strategy'] - df['just_line'])**2)
    sqrt_error = round(df['sqrt_error'].mean()/df['strategy'].iloc[-1], 3)
    sharpe = sharpe_multiplier * df['return'].mean()/df['return'].std()
    sorotino = sharpe_multiplier * sortino_ratio(df['return'].dropna().to_list())
    omega = omega_ratio(df['return'].dropna().to_list())
    dom_ret = calculate_dominant(df['return'].dropna().to_list(), num_ranges=10)
    mean_return = sharpe_multiplier * np.mean(df['return'].dropna().to_list())
    result = round(sharpe * omega * 100, 2)
    final = int(result * (1-sqrt_error))

    density_factor = round(density/(d_buy+d_sell), 2)

    if print_:
        print()
        print("Signals density:     ", density)
        print("Final result:        ", round(df['strategy'].mean() +
                                             df['strategy'].iloc[-1], 2))
        print("Sharpe ratio:        ", round(sharpe, 2))
        print("Sorotino ratio:      ", round(sorotino, 2))
        print("Omega ratio:         ", round(omega, 4))
        print("Dominant return [%]: ", round(dom_ret*100, 5))
        print("Mean return [%]:     ", round(mean_return, 2))
        print("Median return [%]:   ", round(
            sharpe_multiplier * np.median(df['return'].dropna().to_list()), 2))
        print("Sqrt error:          ", sqrt_error)
        print("Density equality:    ", density_factor)
        print("Final:               ", final)

    density_status = True if (density_factor <= 2 and density_factor >= 0.7) else False
    if (not density_status) or (sharpe<sharpe_limit): #or (final < 0) :
        return 0, "NO", density, 1, sqrt_error, final, 0, 0

    if (omega > omega_limit and sharpe > sharpe_limit and final > 0):
        status = "YES"
        print("OK "*30)
        print(f"""#### RESULT: {result} ####""")

    status2, strategy_result, drawdown = stats_from_positions_returns(
        df, symbol, sharpe_multiplier, print_, leverage
        )

    def marker(df):
        buy = df[(df['stance'] == 1) & (df['stance'].shift(1) != 1)]
        sell = df[(df['stance'] == -1) & (df['stance'].shift(1) != -1)]
        for idx in buy.index.tolist():
          plt.plot(
              idx, df.loc[idx]["close"],
              "g^", markersize=8
          )
        for idx in sell.index.tolist():
          plt.plot(
              idx, df.loc[idx]["close"],
              "rv", markersize=8
          )

    if chart and status == "YES" and status2 == "YES":
        cross = np.where(((df.stance == 1) &
                    (df.stance.shift(1) != 1)) |
                    ((df.stance == -1) &
                    (df.stance.shift(1) != -1)), 1, 0)

        fig = plt.figure(figsize=(14,10))
        ax1 = plt.subplot(311)
        plt.plot(df.close, c='b')
        marker(df)
        plt.axhline(y=dominant, color='r', linestyle='-')
        ax2 = plt.subplot(312, sharex = ax1)
        plt.plot(df['strategy'])
        plt.plot(df['just_line'])
        plt.title(f"{symbol}_{interval}_{factor}_{result}_{0}_{sqrt_error}_{final}_{status}")
        ax3 = plt.subplot(313)
        plt.plot(strategy_result)
        plt.pause(interval=1)

    final = int(final + (1+drawdown))
    summary_status = "YES" if (status == "YES" and status2 == "YES" and density_status) else "NO"

    def trend_backtest(df, spread_mean, leverage):
        range2 = 50
        a = list(range(2, range2))
        b = list(range(8, range2))
        z = list(product(a, b))
        z = [i for i in z if i[0] < i[1]]
        results = []
        for a, b in tqdm(z):
            dfx = df.copy()
            dfx['stance2'] = function_when_model_not_work(dfx, a, b)
            # dfx['return2'] = np.where(#(dfx['time2'].dt.date == dfx['time2'].dt.date.shift(1)) &
            #                           (dfx['return'] < 0), (dfx.mkt_move * dfx.stance2.shift(1) -\
            #                             (dfx.cross *(spread_mean)/dfx.open))*leverage, 0)
            dfx['return2'] = np.where((dfx['time2'].dt.date == dfx['time2'].dt.date.shift(1)),
                                      (dfx.mkt_move * dfx.stance2.shift(1) -\
                            (dfx.cross *(spread_mean)/dfx.open))*leverage, 0)
            dfx['strategy2'] = (1+dfx['return2']).cumprod() - 1
            sharpe = sharpe_multiplier * dfx['return2'].mean()/dfx['return2'].std()
            omega = omega_ratio(dfx['return2'].dropna().to_list())
            mean_return = sharpe_multiplier * np.mean(dfx['return2'].dropna().to_list())
            result_ = round(((dfx['strategy2'].iloc[-1] + mean_return) / 2)*sharpe*omega, 2)
            results.append((a, b, result_))
        f_result = sorted(results, key=lambda x: x[2], reverse=True)[0]
        print(f"Best ma factors fast={f_result[0]} slow={f_result[1]}")
        return f_result[0], f_result[1]

    ma_factor1 = 0
    ma_factor2 = 0
    if summary_status == "YES":
        ma_factor1, ma_factor2 = trend_backtest(df, spread_mean, leverage)

    #return result, summary_status, density, how_it_grow, sqrt_error, final
    return result, summary_status, density, 1, sqrt_error, final, ma_factor1, ma_factor2


def primal(df, direction, factor):
    factor = factor - 3
    up = (df.close < df.close.shift(-1)) & \
        (df.close < df.close.shift(-round(factor/2))) & \
        (df.close < df.close.shift(-factor))
    down = (df.close > df.close.shift(-1)) & \
        (df.close > df.close.shift(-round(factor/2))) & \
        (df.close > df.close.shift(-factor))
    df['goal'] = np.where(up, 1, np.NaN)
    df['goal'] = np.where(down, -1, df['goal'])
    df['goal'] = df['goal'].ffill()
    if direction == 'buy':
        df['goal'] = np.where(df['goal'] > df['goal'].shift(1), 'yes', 'no')
    elif direction == 'sell':
        df['goal'] = np.where(df['goal'] < df['goal'].shift(1), 'yes', 'no')
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    return df

def linreg_alma(df, direction, factor):
    col1 = df.ta.linreg(length=round(factor/4))
    col2 = df.ta.alma(length=round(factor))
    col2 = col2.shift(-round(factor))
    df['goal'] = np.where(col1 < col2, 1, -1)
    if direction == 'buy':
        df['goal'] = np.where(((col1 < col2) &
            (col1.shift(1) > col2.shift(1))), 'yes', 'no')
    elif direction == 'sell':
        df['goal'] = np.where(((col1 > col2) &
            (col1.shift(1) < col2.shift(1))), 'yes', 'no')
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    return df

def alma_solo(df, direction, factor):
    ma = ta.alma(df['adj'], length=int(factor))
    ma = ma.shift(-int(factor))
    col1 = df['close']
    col2 = ma
    df['goal'] = np.where(col1 < col2, 1, -1)
    if direction == 'buy':
        df['goal'] = np.where(((col1 < col2) &
                    (col1.shift(1) > col2.shift(1))), 'yes', 'no')
    elif direction == 'sell':
        df['goal'] = np.where(((col1 > col2) &
                    (col1.shift(1) < col2.shift(1))), 'yes', 'no')
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    return df

def t3_shift(df, direction, factor):
    ma = ta.t3(df['close'], length=int(factor), a=0.8)
    ma = ma.shift(-int(factor/3))
    col1 = df['close']
    col2 = ma
    df['goal'] = np.where(col1 < col2, 1, -1)
    if direction == 'buy':
        df['goal'] = np.where(((col1 < col2) &
                    (col1.shift(1) > col2.shift(1))), 'yes', 'no')
    elif direction == 'sell':
        df['goal'] = np.where(((col1 > col2) &
                    (col1.shift(1) < col2.shift(1))), 'yes', 'no')
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    return df

def macd_solo(df, direction, factor):
    fast = df.ta.sma(length=round(factor/3))
    slow = df.ta.sma(length=round(factor))
    signal = fast - slow
    macd = ta.sma(signal, length=int(factor/4))
    signal = signal.shift(-round(factor/3))
    col1 = macd
    col2 = signal
    df['goal'] = np.where(signal > macd, 1, -1)
    if direction == 'buy':
        df['goal'] = np.where(((col1 < col2) &
            (col1.shift(1) > col2.shift(1))), 'yes', 'no')
    elif direction == 'sell':
        df['goal'] = np.where(((col1 > col2) &
            (col1.shift(1) < col2.shift(1))), 'yes', 'no')
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    return df


functions = [t3_shift, alma_solo, primal, macd_solo]

def generate_my_models(
        symbols: list, intervals: list, leverage: int, delete_old_models: bool,
        show_results_on_graph: bool=False, print_: bool=True, generate_model:bool=True) -> None:

    super_start_time = time.time()
    combinations = list(product(intervals, symbols, lr_list, ts_list, factors, functions))
    number_of_combinations = len(combinations)
    i = 0
    times = []

    if delete_old_models == "no":
        return ""

    if delete_old_models:
        delete_model(f"{catalog}\\models\\", '')

    results = []
    for interval in tqdm(intervals):
        finals = []
        for symbol in symbols:
            df_raw = get_data_for_model(symbol, interval, 1, bars)
            for factor in factors:
                print("DF length: ", len(df_raw))
                df = data_operations(df_raw.copy(), factor)
                for function in functions: 
                    for t_set in ts_list:
                        train_length = 0.98
                        dataset = df.copy()[:int(train_length*len(df))]
                        testset = df.copy()[int(train_length*len(df)):]
                        for learning_rate in lr_list:
                            parameters = {
                                'learning_rate': learning_rate,
                                'max_depth': 2*len(dataset.columns),
                                'colsample_by*': 0.9,
                                'min_child_weight': int(len(dataset.columns)/17),
                                'subsample': 0.7,
                                'random_state': 42,
                                'eval_metric': 'auc',
                                'tree_method': 'gpu_hist',
                                'device': 'cuda',
                                'objective': 'binary:logistic',
                                'gamma': 1,
                                'alpha': 0.2,
                                'lambda': 0.01,
                            }
                            start = time.time()
                            try:
                                print(f"\nSymbol: {symbol}; Interval: {interval}; Factor: {factor}")
                                print(f"Function: {function.__name__}")
                                models_buy, d_buy = train_dataset(dataset, 'buy', parameters, factor,
                                                        n_estimators, function, t_set, show_results=show_results_on_graph,
                                                        n_splits=n_splits)
                                models_sell, d_sell = train_dataset(dataset, 'sell', parameters, factor,
                                                        n_estimators, function, t_set, show_results=show_results_on_graph,
                                                        n_splits=n_splits)
                                for m in range(len(models_buy)):
                                    dfx = testset.copy()
                                    buy = models_buy[m].predict(xgb.DMatrix(testset))
                                    sell = models_sell[m].predict(xgb.DMatrix(testset))
                                    buy = np.where(buy > probability_edge, 1, 0)
                                    sell = np.where(sell > probability_edge, -1, 0)
                                    dfx['time2'] = pd.to_datetime(dfx['time'], unit='s')
                                    dfx['stance'] = buy + sell
                                    dfx['stance'] = dfx['stance'].replace(0, np.NaN)
                                    dfx['stance'] = np.where((dfx['time2'].dt.hour > morning_hour) & (dfx['time2'].dt.hour < evening_hour), dfx['stance'], np.NaN)
                                    dfx['stance'] = dfx.groupby(dfx['time2'].dt.date)['stance'].ffill()
                                    dfx['stance'] = dfx['stance'].replace(np.NaN, 0)
                                    dfx = dfx[dfx['stance'] != 0]

                                    for market in ['e', 'u']:
                                        dfx_market = dfx.copy()
                                        if market == 'e':
                                            dfx_market = dfx_market[dfx_market['time2'].dt.hour <= change_hour + 2]
                                        else:
                                            dfx_market = dfx_market[dfx_market['time2'].dt.hour >= change_hour]
                                        dfx_market.reset_index(drop=True, inplace=True)
                                        result, status, density, how_it_grow, sqrt_error, final, ma_factor1, ma_factor2 = \
                                            strategy_with_chart_(
                                                d_buy, d_sell, dfx_market, leverage, interval, symbol, factor, chart=show_results_on_graph, print_=print_
                                                                )
                                        results.append((symbol, interval, leverage, factor, result, density,
                                                        how_it_grow, sqrt_error, final, status))
                                        if generate_model and ma_factor1 != 0:
                                            if final in finals:
                                                continue
                                            _lr_name = str(learning_rate).split('.')[-1]
                                            _ts_name = str(t_set).split('.')[-1]
                                            #name_ = f'{market}_{function.__name__[0]}_{_lr_name}_{_ts_name}_{symbol}_{interval}_{factor}_{final}'
                                            name_ = f'{market}_{function.__name__[0]}_{ma_factor1}_{ma_factor2}_{_lr_name}_{_ts_name}_{symbol}_{interval}_{factor}_{final}'
                                            finals.append(final)
                                            models_buy[m].save_model(f"{catalog}\\models\\{name_}_buy.model") # models_buy[-1]
                                            models_sell[m].save_model(f"{catalog}\\models\\{name_}_sell.model") # models_sell[-1]
                            except Exception as e:
                                print(e)
                                i += 1
                                continue
                            end = time.time()
                            times.append(end-start)
                            i += 1
                            time_remaining = round((number_of_combinations - i) * np.mean(times), 2)
                            time_info(time_remaining, 'Time remaining')

    super_end_time = time.time()
    total_duration = super_end_time - super_start_time
    time_info(total_duration, 'Total duration')


if __name__ == '__main__':
    from config.parameters import symbols
    generate_my_models(symbols, intervals, leverage, False, True)