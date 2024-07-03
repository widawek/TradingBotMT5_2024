import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import MetaTrader5 as mt
import pandas_ta as ta
import matplotlib.pyplot as plt
from tqdm import tqdm
from functions import *
import warnings
warnings.filterwarnings("ignore")
mt.initialize()
import os
catalog = os.path.dirname(__file__)

morning_hour = 6
evening_hour = 22


def stats_from_positions_returns(df, symbol, sharpe_multiplier, print_, leverage):
    status = "NO"
    annotation = "clean position"
    df['cross'] = np.where(df['stance'] != df['stance'].shift(1), 1, 0)
    density = round((df['cross'].sum()/len(df))*100, 2)
    returns, _, _ = get_returns(df, symbol)
    strategy_result = (1 + returns).cumprod() - 1
    curve_result1 = is_this_curve_grow(strategy_result.dropna(), density)
    curve_result2 = is_this_curve_grow2(strategy_result.dropna(), density)
    curve_result3 = is_this_curve_grow3(strategy_result.dropna(), density)
    how_it_grow = round((curve_result1 * curve_result2 * curve_result3/leverage)*
                        sharpe_multiplier, 2)
    
    sharpe = sharpe_multiplier * returns.mean()/returns.std()
    sorotino = sharpe_multiplier * sortino_ratio(returns.to_list())
    omega = omega_ratio(returns.to_list())
    dom_ret = calculate_dominant(returns.to_list(), num_ranges=10)
    mean_return = sharpe_multiplier * np.mean(returns.to_list())
    result = round((((sharpe + sorotino)/2) * omega * mean_return) * 100, 2)

    if print_:
        print()
        print(f"Signals density {annotation}:      ", density)
        print(f"Final result {annotation}:         ", round(df['strategy'].mean() +
                                             df['strategy'].iloc[-1], 2))
        print(f"Sharpe ratio {annotation}:         ", round(sharpe, 2))
        print(f"Sorotino ratio {annotation}:       ", round(sorotino, 2))
        print(f"Omega ratio {annotation}:          ", round(omega, 4))
        print(f"Dominant return [%] {annotation}:  ", round(dom_ret*100, 5))
        print(f"Mean return [%] {annotation}:      ", round(mean_return, 2))
        print(f"Median return [%] {annotation}:    ", round(
            sharpe_multiplier * np.median(df['return'].dropna().to_list()), 2))
        print(f"Growing factor {annotation}:       ", how_it_grow)

    if (omega > 1 and sharpe > 3 and sorotino > 3 and mean_return > 0 and
     (how_it_grow > 10 or how_it_grow == np.inf) and curve_result3 > 1 and
     strategy_result.iloc[-1] > 0):
        status = "YES"
        print(f"OK {annotation} "*30)
        print(f"""#### RESULT: {result} ####""")
    return status, strategy_result


def data_operations(df):
    df['adj'] = (df.high + df.low + df.close) / 3
    df['adj_higher'] = np.where(df['adj'] > df['adj'].shift(1), 1, 0)
    df['high_higher'] = np.where(df['high'] > df['high'].shift(1), 1, 0)
    df['low_lower'] = np.where(df['low'] < df['low'].shift(1), 1, 0)
    df['close_higher'] = np.where(df['close'] > df['close'].shift(1), 1, 0)
    df['volume_square'] = np.sin(np.log(df['volume']**2))
    df['high_square'] = np.sin(np.log(df['high']**2))
    df['low_square'] = np.sin(np.log(df['low']**2))
    df['close_square'] = np.sin(np.log(df['close']**2))
    df['high_close'] = df['high'] - df['close']
    df['low_close'] = df['low'] - df['close']
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
    #df['vola_vol_log'] = np.log(df['vola_vol']/df['vola_vol'].shift(1))
    df.replace(np.inf, 0, inplace=True)
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    return df


def interval_time_sharpe(interval):
    if interval[0] == 'D':
        return np.sqrt(252)
    elif interval[0] == 'H':
        return np.sqrt(252 * 24 / int(interval[1:]))
    elif interval[0] == 'M':
        return np.sqrt(252 * 24 * 60 / int(interval[1:]))


def ma_shift4(df, direction, factor):
    ma = ta.sma(df['close'], length=int(factor))
    ma = ma.shift(-int(factor/2))
    df['goal'] = np.where(df['close'] < ma, 1, -1)
    if direction == 'buy':
        df['goal'] = np.where(((df['close'] < ma) &
            (df['close'].shift(1) > ma.shift(1))), 'yes', 'no')
    elif direction == 'sell':
        df['goal'] = np.where(((df['close'] > ma) &
            (df['close'].shift(1) < ma.shift(1))), 'yes', 'no')
    #df['goal'] = df['goal'].shift(-int(factor/2))
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    return df


def train_dataset(df, direction, parameters, factor, n_estimators, function, t_set, show_results=True):
    dataset = df.copy()
    dataset = function(dataset, direction, factor)
    #isolate the x and y variables
    y = dataset.iloc[:, -1].values
    X = dataset._get_numeric_data()

    #isolate the categorical variables
    dataset_categorical = dataset.select_dtypes(exclude = "number")

    #transform categorical variables into dummy variables
    dataset_categorical = pd.get_dummies(data = dataset_categorical,
                                        drop_first = True)

    #joining numerical and categorical datasets
    final_dataset = pd.concat([X, dataset_categorical], axis = 1)
    feature_columns = list(final_dataset.columns.values)
    feature_columns = feature_columns[:-1]

    #isolate the x and y variables part 2
    y = final_dataset.iloc[:, -1].values
    X = final_dataset.iloc[:, :-1].values

    #split dataset into training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=t_set,
                                                        random_state=1502)

    print(f"\nGoal: {round(np.mean(y_test)*100, 2)} % yes/no")
    #create xgboost matrices part 2
    Train = xgb.DMatrix(X_train, label = y_train, feature_names = feature_columns)
    Test = xgb.DMatrix(X_test, label = y_test, feature_names = feature_columns)

    model = xgb.train(
        params = parameters,
        dtrain = Train,
        num_boost_round = n_estimators,
        evals = [(Test, "Yes")],
        early_stopping_rounds=40,
        verbose_eval = 10000
        )

    if show_results:
        #predictions
        predictions2 = model.predict(Test)
        predictions2 = np.where(predictions2 > 0.5, 1, 0)
        #confusion matrix
        confusion_matrix2 = confusion_matrix(y_test, predictions2)
        print(confusion_matrix2)
        report2 = classification_report(y_test, predictions2)
        print(report2)
        # plot importances
        # xgb.plot_importance(model, max_num_features=10)
    return model


def strategy_with_chart_(df, leverage, interval, symbol, factor, chart=True, print_=True):
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
                            df.mkt_move * df.stance.shift(1) * leverage -\
                                (df.cross *(df.spread/divider)/df.open), 0)
    df['strategy'] = (1+df['return']).cumprod() - 1
    df['max_price'] = df.apply(lambda row: max_vol_times_price_price(df.loc[:row.name]), axis=1)
    dominant = calculate_dominant(df['close'].to_list(), num_ranges=10)
    just_line = np.linspace(df['strategy'].dropna().iloc[0], df['strategy'].dropna().iloc[-1], num=len(df['strategy'].dropna()))
    df.loc[df.index[-1] - len(df.strategy.dropna()) + 1:df.index[-1] , 'just_line'] = just_line
    
    df['sqrt_error'] = np.sqrt((df['strategy'] - df['just_line'])**2)
    sqrt_error = round(df['sqrt_error'].mean()/df['strategy'].iloc[-1], 3)
    
    curve_result1 = is_this_curve_grow(df['strategy'].dropna(), density)
    curve_result2 = is_this_curve_grow2(df['strategy'].dropna(), density)
    curve_result3 = is_this_curve_grow3(df['strategy'].dropna(), density)
    how_it_grow = round((curve_result1 * curve_result2 * curve_result3/leverage)*
                        sharpe_multiplier, 2)
    
    sharpe = sharpe_multiplier * df['return'].mean()/df['return'].std()
    sorotino = sharpe_multiplier * sortino_ratio(df['return'].dropna().to_list())
    omega = omega_ratio(df['return'].dropna().to_list())
    dom_ret = calculate_dominant(df['return'].dropna().to_list(), num_ranges=10)
    mean_return = sharpe_multiplier * np.mean(df['return'].dropna().to_list())
    result = round((((sharpe + sorotino)/2) * omega * mean_return) * 100, 2)
    final = int(result * (how_it_grow if how_it_grow != np.inf else 1000) * (1-sqrt_error))
    
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
        print("Growing factor:      ", how_it_grow)
        print("Sqrt error:          ", sqrt_error)
        print("Final:               ", final)

    if (omega > 1 and sharpe > 3 and sorotino > 3 and mean_return > 0 and
     (how_it_grow > 10 or how_it_grow == np.inf) and curve_result3 > 1 and
      df['strategy'].iloc[-1] > 0):
        status = "YES"
        print("OK "*30)
        print(f"""#### RESULT: {result} ####""")
    
    status2, strategy_result = stats_from_positions_returns(
        df, symbol, sharpe_multiplier, print_, leverage
        )
    
    if chart and status == "YES" and status2 == "YES":
        fig = plt.figure(figsize=(30,12))
        ax1 = plt.subplot(311)
        plt.plot(df.close, c='b')
        plt.axhline(y=dominant, color='r', linestyle='-')
        ax2 = plt.subplot(312, sharex = ax1)
        plt.plot(df['strategy'])
        plt.plot(df['just_line'])
        plt.title(f"{symbol}_{interval}_{factor}_{result}_{how_it_grow}_{sqrt_error}_{final}_{status}")
        ax3 = plt.subplot(313)
        plt.plot(strategy_result)
        plt.show()
    
    summary_status = "YES" if (status == "YES" and status2 == "YES") else "NO"
    return result, summary_status, density, how_it_grow, sqrt_error, final


def generate_my_models(
        symbols, intervals, leverage, delete_old_models, 
        show_results_on_graph=False, print_=True, generate_model=True):

    if delete_old_models:
        delete_model(f"{catalog}\\models\\", '')

    n_estimators = 4000
    function = ma_shift4
    lr_list = [0.4, 0.5, 0.6, 0.7, 0.8]
    ts_list = [0.2, 0.4]
    factors = [_ for _ in range(4, 31, 2)]

    results = []
    for interval in tqdm(intervals):
        for symbol in symbols:
            df = get_data_for_model(symbol, interval, 1, 80000)
            #df = my_test
            print("DF length: ", len(df))
            df = data_operations(df)
            train_length = 0.98
            if interval == "M1" or interval == "M2":
                train_length = 0.97
            dataset = df.copy()[:int(train_length*len(df))]
            testset = df.copy()[int(train_length*len(df)):]
            for learning_rate in lr_list:
                parameters = {
                    'learning_rate': learning_rate,
                    'max_depth': len(dataset.columns),
                    'colsample_by*': 0.9,
                    'min_child_weight': len(dataset.columns),
                    'random_state': 42,
                    'eval_metric': 'auc',
                    'objective': 'binary:hinge'
                }
                for t_set in ts_list:
                    for factor in factors:
                        print("Interval: ", interval)
                        dfx = testset.copy()
                        model_buy = train_dataset(dataset, 'buy', parameters, factor,
                                                n_estimators, function, t_set, show_results=False)
                        model_sell = train_dataset(dataset, 'sell', parameters, factor,
                                                n_estimators, function, t_set, show_results=False)
                        buy = model_buy.predict(xgb.DMatrix(testset))
                        sell = model_sell.predict(xgb.DMatrix(testset))
                        buy = np.where(buy > 0, 1, 0)
                        sell = np.where(sell > 0, -1, 0)
                        dfx['time2'] = pd.to_datetime(dfx['time'], unit='s')
                        dfx['stance'] = buy + sell
                        dfx['stance'] = dfx['stance'].replace(0, np.NaN)
                        dfx['stance'] = dfx['stance'].ffill()
                        dfx['stance'] = np.where((dfx['time2'].dt.hour > morning_hour) & (dfx['time2'].dt.hour < evening_hour), dfx['stance'], 0)
                        dfx = dfx[dfx['stance'] != 0]
                        dfx.reset_index(drop=True, inplace=True)
                        result, status, density, how_it_grow, sqrt_error, final = strategy_with_chart_(
                            dfx, leverage, interval, symbol, factor, chart=show_results_on_graph, print_=print_
                            )
                        results.append((symbol, interval, leverage, factor, result, density,
                                        how_it_grow, sqrt_error, final, status))
                        if generate_model:
                            if status == "YES":
                                model_buy.save_model(f"{catalog}\\models\\{symbol}_{interval}_{factor}_{final}_buy.model")
                                model_sell.save_model(f"{catalog}\\models\\{symbol}_{interval}_{factor}_{final}_sell.model")


if __name__ == '__main__':
    generate_my_models(['GBPUSD'], ['M5', 'M10', 'M20'], 20, False, True, True, False)