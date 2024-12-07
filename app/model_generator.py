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
from app.mg_functions import *

# parameters
def params(learning_rate, df_cols):
    params = {
        'learning_rate':        learning_rate,
        'max_depth':            2*len(df_cols),
        'colsample_by*':        0.9,
        'min_child_weight':     int(len(df_cols)/17),
        'subsample':            0.7,
        'random_state':         42,
        'eval_metric':          'auc',
        'tree_method':          'gpu_hist',
        'device':               'cuda',
        'objective':            'binary:logistic',
        'gamma':                1,
        'alpha':                0.2,
        'lambda':               0.01,
        }
    return params                          

functions = [t3_shift, alma_solo, primal, macd_solo]
train_length = 0.975


# functions
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

    if (omega > omega_limit and sharpe > sharpe_limit and kk > kk_limit and dom_ret > drawdown):
        status = "YES"
        print(f"OK {annotation} "*30)
        print(f"""#### RESULT: {result} ####""")
    return status, strategy_result, drawdown


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

    # def trend_backtest(df, spread_mean, leverage):
    #     range2 = 50
    #     a = list(range(2, range2))
    #     b = list(range(8, range2))
    #     z = list(product(a, b))
    #     z = [i for i in z if i[0] < i[1]]
    #     results = []
    #     for a, b in tqdm(z):
    #         dfx = df.copy()[int(-len(df)/2):]
    #         dfx['stance2'] = function_when_model_not_work(dfx, a, b)
    #         # dfx['return2'] = np.where(#(dfx['time2'].dt.date == dfx['time2'].dt.date.shift(1)) &
    #         #                           (dfx['return'] < 0), (dfx.mkt_move * dfx.stance2.shift(1) -\
    #         #                             (dfx.cross *(spread_mean)/dfx.open))*leverage, 0)
    #         dfx['return2'] = np.where((dfx['time2'].dt.date == dfx['time2'].dt.date.shift(1)),
    #                                   (dfx.mkt_move * dfx.stance2.shift(1) -\
    #                         (dfx.cross *(spread_mean)/dfx.open))*leverage, 0)
    #         dfx['strategy2'] = (1+dfx['return2']).cumprod() - 1
    #         sharpe = sharpe_multiplier * dfx['return2'].mean()/dfx['return2'].std()
    #         omega = omega_ratio(dfx['return2'].dropna().to_list())
    #         #mean_return = sharpe_multiplier * np.mean(dfx['return2'].dropna().to_list())
    #         #result_ = round(((dfx['strategy2'].iloc[-1] + mean_return) / 2)*sharpe*omega, 2)
    #         result_ = round(sharpe*omega, 2)
    #         results.append((a, b, result_))
    #     f_result = sorted(results, key=lambda x: x[2], reverse=True)[0]
    #     print(f"Best ma factors fast={f_result[0]} slow={f_result[1]}")
    #     return f_result[0], f_result[1]

    ma_factor1 = 0
    ma_factor2 = 0
    # if summary_status == "YES":
    #     ma_factor1, ma_factor2 = trend_backtest(df, spread_mean, leverage)

    #return result, summary_status, density, how_it_grow, sqrt_error, final
    return result, summary_status, density, 1, sqrt_error, final, ma_factor1, ma_factor2


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
            from config.parameters import leverage
            if symbol in ['BTCUSD', 'DE40', 'EURCAD', 'ETHUSD']:
                leverage = 2
            df_raw = get_data_for_model(symbol, interval, 1, bars)
            for factor in factors:
                print("\nDF length: ", len(df_raw))
                df = data_operations(df_raw.copy(), factor)
                for function in functions:
                    for t_set in ts_list:
                        dataset = df.copy()[:int(train_length*len(df))]
                        testset = df.copy()[int(train_length*len(df)):]
                        for learning_rate in lr_list:
                            parameters = params(learning_rate, dataset.columns)
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