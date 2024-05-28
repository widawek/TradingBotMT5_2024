import pandas as pd
import os
from sklearn.model_selection import train_test_split # , learning_curve
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import MetaTrader5 as mt
import warnings
import pandas as pd
from xgboost import XGBRegressor
import numpy as np
from tqdm import tqdm
warnings.filterwarnings("ignore")
mt.initialize()
catalog = os.path.dirname(__file__)


def pandas_options():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)


def round_number_(symbol):
    return mt.symbol_info(symbol).digits


def get_returns(df, symbol):
    r_num = round_number_(symbol)
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    df['cross'].iloc[-1] = 1
    ret = df[df['cross'] == 1][['time', 'close', 'position', 'strategy']]
    ret.reset_index(drop=True, inplace=True)
    ret['t_delta'] = ret['time'] - ret['time'].shift(1)
    time_list = [int((i.total_seconds()/60)) for i in
                 ret['t_delta'].dropna()]
    try:
        mean_time = round(sum(time_list)/len(time_list))
    except ZeroDivisionError:
        mean_time = 0
    ret = ret.replace(np.NaN, 0)
    ret.reset_index(drop=True, inplace=True)
    ret['return'] = ret['strategy'] - ret['strategy'].shift(1)
    positive = ret[ret['return'] > 0]
    positive['positive'] = positive['return'] * positive['close']
    negative = ret[ret['return'] < 0]
    negative['negative'] = abs(negative['return']) * negative['close']
    tp = positive['positive'].mean() + positive['positive'].std()
    tp = round(tp, r_num)
    sl = negative['negative'].mean() + negative['negative'].std()
    sl = round(sl, r_num)
    if sl == 0:
        sl = round(0.5*tp, r_num)
    return ret['return'].dropna(), mean_time, tp, sl


def sorotino_ratio(returns):
    mean = np.mean(returns)
    ret_count = len(returns[returns < 0])
    if ret_count == 0:
        return 10
    elif ret_count == 1:
        return 10
    stdNeg = returns[returns < 0].std()
    return mean/stdNeg


def sharpe_ratio(returns):
    mean = np.mean(returns)
    stdNeg = returns.std()
    return mean/stdNeg


def average_drawdown(returns):
    drawdowns = []
    drawdown = 0
    for r in returns:
        if r < 0:
            drawdown += abs(r)
        else:
            if drawdown > 0:
                drawdowns.append(drawdown)
                drawdown = 0
    averageDD = np.mean(drawdowns) if len(drawdowns) > 0 else 0
    return averageDD


def max_drawdown(returns):
    compRet = (returns+1).cumprod()
    peak = compRet.expanding(min_periods=1).max()
    dd = (compRet/peak)-1
    return dd.min()


def kelly_criterion(returns):
    good = [i for i in returns if i > 0]
    bad = [i for i in returns if i < 0]
    p = len(good) / len(returns)
    q = len(bad) / len(returns)
    a = abs(sum(bad))
    b = sum(good)
    try:
        win = (p/a)
    except ZeroDivisionError:
        return 0.1
    try:
        loss = (q/b)
    except ZeroDivisionError:
        return -0.1
    kk = win - loss
    if kk > 20:
        kk = 20
    if kk < -20:
        kk = -20
    return kk


def omega_ratio(returns):
    try:
        L = np.percentile(returns, 50)
        omega_up = sum(np.maximum(0, returns - L)) / len(returns)
        omega_down = sum(np.maximum(0, L - returns)) / len(returns)
        omega_ratio = round(omega_up / omega_down if omega_down != 0 else 100, 3)
    except IndexError:
        return 0
    return omega_ratio


def timeframe_(tf):
    timeframe = getattr(mt, 'TIMEFRAME_{}'.format(tf))
    return timeframe


def copy_shifted(df, factor):
    if not factor:
        return df
    for column in list(df.columns):
        if not 'date' in column:
            for i in range(1, factor+1):
                df[f"{column}_{i}"] = df[column].shift(i)
    return df

def get_data(symbol, tf, start, counter):
    data = pd.DataFrame(mt.copy_rates_from_pos(
                        symbol, timeframe_(tf), start, counter))
    data["time"] = pd.to_datetime(data["time"], unit="s")
    data = data.drop(["real_volume"], axis=1)
    data.columns = ["time", "open", "high", "low",
                    "close", "volume", "spread"]
    return data


def get_data_price(symbol, tf, start, counter):
    data = pd.DataFrame(mt.copy_rates_from_pos(
                        symbol, timeframe_(tf), start, counter))
    data["time"] = data["time"].astype(int)
    data = data.drop(["real_volume", "spread"], axis=1)
    data.columns = ["time", "open", "high", "low",
                    "close", "volume"]
    return data


def get_df(factor, df_divider, symbol, train=True):
    zzz = 14
    df_raw = pd.read_excel(f'{catalog}\\{symbol}_ml_data.xlsx', index_col=0)
    print("DataFrame length", len(df_raw))
    df_raw['week_day'] = df_raw.date.dt.weekday
    df_raw['standard_dev_c'] = df_raw['price_close'].rolling(zzz).std()
    df = df_raw.copy()
    from_ = int(len(df)*df_divider)
    if train:
        df = df.copy()[:from_]
        df = copy_shifted(df, factor)
        df['target_close'] = df['price_close'].shift(-factor)
        df['target_high'] = df['price_high'].shift(-factor)
        df['target_low'] = df['price_low'].shift(-factor)
    else:
        df = df.copy()[from_-factor:]
        df = copy_shifted(df, factor)
        
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    return df.drop(columns=['date']), df['date']


def get_df_2(factor, df_divider, symbol, train=True):
    zzz = 14
    
    def higher_lower_closer(df, factor):
        df['high_higher'] = np.where(df['high'] > df['high'].shift(factor), 1, 0)
        df['high_lower'] = np.where(df['low'] < df['low'].shift(factor), 1, 0)
        df['close_closer'] = np.where(df['close'] > df['close'].shift(factor), 1, -1)
        return df
    
    years_ = 5
    df_raw = get_data_price(symbol, "D1", 1, int(365*years_))
    df_raw['standard_dev_c'] = df_raw['close'].rolling(zzz).std()
    df_raw['week_day'] = pd.to_datetime(df_raw.time, unit='s').dt.weekday
    df = df_raw.drop(columns=['time'])
    df = df_raw.copy()
    from_ = int(len(df)*df_divider)
    if factor == 0:
        factor_ = 1
    else:
        factor_ = factor

    if train:
        df = df.copy()[:from_]
        df = copy_shifted(df, factor)
        df = higher_lower_closer(df, factor)
        
        df['target_close'] = df['close'].shift(-factor_)
        df['target_high'] = df['high'].shift(-factor_)
        df['target_low'] = df['low'].shift(-factor_)
    else:
        df = df.copy()[from_-factor:]
        df = copy_shifted(df, factor)
        df = higher_lower_closer(df, factor)
    df = df.dropna()
    df.reset_index(drop=True, inplace=True)
    return df

def gimme_returns(mdf, leverage):
    mdf['position'] = mdf['stance'].ffill()
    mdf["cross"] = np.where(
        ((mdf.position == 1) & (mdf.position.shift(1) != 1)) | 
        ((mdf.position == -1) & (mdf.position.shift(1) != -1)), 1, 0
    )
    mdf['mkt_move'] = np.log(mdf.close / mdf.close.shift(1))
    mdf['return'] = mdf.mkt_move * mdf.position.shift(1) * leverage
    return mdf


def conditional_return(df, interval, value_in_days, leverage):
    interval = int(interval[1:])
    buffer = int((1440/interval) * value_in_days)
    df['stance'] = np.where((df['counter_column'] > buffer) &
                            (df['position_return'] < 0) &
                            (df['position'] == -1),
                            1, df['stance'])
    df['stance'] = np.where((df['counter_column'] > buffer) &
                            (df['position_return'] < 0) &
                            (df['position'] == 1),
                            -1, df['stance'])
    return gimme_returns(df, leverage)


def train_model_natural(factor, symbol, params, targets, df_divider):
    t_s = 0.3
    df_raw = get_df_2(factor, df_divider, symbol)
    cols = [i for i in df_raw.columns if not 'target' in i]
    dfx = df_raw.copy()
    X = dfx[cols]
    models = []
    for target in targets:
        y = dfx[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=t_s, random_state=80, shuffle=True)
        model = XGBRegressor(**params)
        # model.fit(X_train, y_train,
        #           eval_metric=["merror", "mlogloss"],
        #           verbose=True
        #           )

        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            eval_metric='rmse',
            early_stopping_rounds=20,
            verbose=0
        )

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        xxx = X_test.close - y_pred
        sharpe = round((X_test.close.mean()/xxx.std())/100, 2)
        percent_absolute_error = round(mae*100/X[target[7:]].mean(), 2)
        standard_dev = round(X[target[7:]].std()/X[target[7:]].mean(), 2)
        if percent_absolute_error < standard_dev:
            print("OK")
        
        models.append(model)
    return models, df_raw, sharpe



def strategy_with_chart_3(mdf, leverage, symbol):
    mdf = gimme_returns(mdf, leverage)
    mdf['strategy'] = initial_capital * (1 + mdf['return']).cumprod() - initial_capital
    mdf = mdf[mdf['strategy'] != np.NaN][1:]
    mdf.reset_index(drop=True, inplace=True)
    # Identify buy and sell signals
    x = mdf['close'].shift(1) - mdf['model_close']
    sharpe_for_close = (mdf['close'].mean() / x.std())/100
    
    x = mdf['high'].shift(1) - mdf['model_high']
    sharpe_for_high = (mdf['high'].mean() / x.std())/100
    
    x = mdf['low'].shift(1) - mdf['model_low']
    sharpe_for_low = (mdf['low'].mean() / x.std())/100
    
    sharpe_all = round(np.mean([sharpe_for_close, sharpe_for_high, sharpe_for_low]), 2)
    # Plotting the charts
    returns, mean_time, tp, sl = get_returns(mdf, symbol)
    if len(returns) -1 > 0:
        real_sharpe = sharpe_ratio(returns)
        omega = omega_ratio(returns)
        sorotino = sorotino_ratio(returns)
        kelly = kelly_criterion(returns)
        x = [True if i > 0 else False for i in
             [real_sharpe, omega, sorotino, kelly, sharpe_all,
              mdf['strategy'].iloc[-1]]
             ]
        if all(x):
            sha = f'Sharpe: {round(real_sharpe*100, 2)} '
            ome = f'Omega: {omega} '
            sor = f'Sorotino: {round(sorotino*100,2)} '
            kel = f'Kelly: {round(kelly*100, 2)} '
            print(sha, '\n', ome, '\n', sor, '\n', kel, '\n')   
    strategy_result = round((mdf['strategy'].mean() +
                             mdf['strategy'].iloc[-1]) / 2, 2)
    return strategy_result, sharpe_all, mdf['position'].iloc[-1]



def condition(data):
    cond_long = (((data['close'] < data['price_long']) & (data['close'].shift(1) > data['price_long'].shift(1))) |\
                 ((data['close'] < data['price_long']) & (data['time'].dt.date != data['time'].dt.date.shift(1))))
    cond_short = (((data['close'] > data['price_short']) & (data['high'].shift(1) < data['price_short'].shift(1))) |\
                  ((data['close'] > data['price_short']) & (data['time'].dt.date != data['time'].dt.date.shift(1))))
    cond_long = cond_long & (data['last_price_close'] < data['model_close'])
    cond_short = cond_short & (data['last_price_close'] > data['model_close'])
    return cond_long, cond_short


def this_shit(symbol, factor, params, leverage, mode, buffer, interval,
              df_divider):

    models, _, _ = train_model_natural(factor, symbol, params,
                                ['target_close',
                                'target_high',
                                'target_low'], df_divider)

    df = get_df_2(factor, df_divider, symbol, train=False)
    
    cols = [i for i in df.columns if not 'target' in i]
    df = df[cols]
    dfx = df.copy()
    dfx['model_close'] = models[0].predict(df)
    dfx['model_high'] = models[1].predict(df)
    dfx['model_low'] = models[2].predict(df)
    dfx['model_close'] = dfx['model_close'].shift(1)
    dfx['model_high'] = dfx['model_high'].shift(1)
    dfx['model_low'] = dfx['model_low'].shift(1)
    dfx['model_close'] = dfx['model_close'].astype(float)
    dfx['model_high'] = dfx['model_high'].astype(float)
    dfx['model_low'] = dfx['model_low'].astype(float)
    if mode == 'natural':
        dfx['time'] = pd.to_datetime(dfx['time'], unit='s')
    dfx['vola_'] = dfx.high - dfx.low
    y = (dfx['model_high']-dfx['high'])/dfx['vola_']
    z = (dfx['model_low']-dfx['low'])/dfx['vola_']
    w = (dfx['model_high']-dfx['low'])/dfx['vola_']
    buffer_value = round((abs(y).mean()+abs(z).mean())/2, 3)

    dfz = dfx.copy()[['time', 'close', 'high', 'low', 'model_high', 'model_low', 'model_close']]
    dfz = dfz.rename(columns={'close': 'price_close', 'high': 'price_high', 'low': 'price_low'})
    dfz['last_model_close'] = dfz['model_close'].shift(1)
    dfz['last_price_close'] = dfz['price_close'].shift(1)
    if buffer == 'on':
        dfz['volatility'] = abs(dfz['model_high'] - dfz['model_low'])
        dfz['volatility_'] = dfz['volatility'].rolling(window=5).mean()
        df['buffer'] = dfz['volatility'] * buffer_value/w
        dfz['price_long'] = dfz['model_low'] + df['buffer']
        dfz['price_short'] = dfz['model_high'] - df['buffer']
    elif buffer == 'off':
        dfz['price_long'] = dfz['model_low']
        dfz['price_short'] = dfz['model_high']
    elif isinstance(buffer, int) or isinstance(buffer, float):
        dfz['volatility'] = abs(dfz['model_high'] - dfz['model_low'])
        dfz['volatility_'] = dfz['volatility'].rolling(window=5).mean()
        df['buffer'] = dfz['volatility'] * buffer_value
        dfz['price_long'] = dfz['model_low'] + df['buffer']
        dfz['price_short'] = dfz['model_high'] - df['buffer']
    dfz['price_long'] = dfz['price_long'].astype(float)
    dfz['price_short'] = dfz['price_short'].astype(float)
    
    dfz = dfz.dropna()
    df = get_data(symbol, interval, 0, 50000)
    data = pd.merge_asof(df, dfz, on='time')
    data = data.dropna()
    
    cond_long, cond_short = condition(data)

    data['stance'] = np.where(cond_long, 1, np.NaN)
    data['stance'] = np.where(cond_short, -1, data['stance'])
    result, sharpe_for_close, position = strategy_with_chart_3(
        data, leverage, symbol)
    return result, sharpe_for_close, position, models

global initial_capital 
initial_capital = 500
leverage = 10
factor = 1
mode = 'natural'
buffer = 'on'
interval = 'M3'

params = {
    'n_estimators': 300,
    'learning_rate': 0.6,
    'tree_method': 'gpu_hist',
    'device': 'cuda',
    'predictor': 'gpu_predictor',
    'subsample': 0.9,
    'colsample_by*': 0.8,
    'num_parallel_tree': 20, 
    'min_child_weight': 12,
    'max_depth': 20,
    'objective': 'reg:squarederror',
}

symbols_ = ['JP225', 'USTEC', 'UK100', 'DE40', 'US30', 'AUDCAD', 'AUDUSD',
           'BTCUSD', 'AUDNZD', 'USDJPY', 'USDCAD', 'XAGAUD', 'XAGUSD', 'EURJPY',
           'NZDCAD', 'EURUSD', 'USDCHF', 'GBPUSD', 'GBPJPY', 'XAUUSD']


def today_position():
    best_models = []
    range_ = range(97, 99)
    for d in tqdm(range_):
        for buffer in ['on', 'off', 0.1, 0.3]:
            for symbol in symbols_:
                df_divider = float(d/100)
                result, sharpe_for_close, position, models = this_shit(
                    symbol, factor, params, leverage, mode, buffer,
                    interval, df_divider)
                best_models.append((symbol, df_divider, buffer, result,
                                    sharpe_for_close, position, models))

    pandas_options()
    res = pd.DataFrame(best_models, columns=[
        'symbol', 'divider', 'buffer_mode', 'result', 'sharpe', 'position', 'models'])
    res = res[['symbol', 'divider', 'buffer_mode', 'result', 'position']]
    res = res[(res['result'] > 0)]
    res = res.sort_values(by=['symbol', 'result'], ascending=False)
    res['metric'] = res['position'] * res['result']
    final = res.groupby('symbol').agg({'metric': 'mean', 'symbol': 'count'})
    final['actual_pos'] = np.where(final['metric'] > 0, 0, 1)
    final = final.rename(columns={'symbol': 'counter'})
    final = final[final['counter'] > 1]
    print(final)
    symbols = [(s, r) for s, r in zip(final.index.to_list(), final['actual_pos'].to_list())]
    print(symbols)
    return symbols


if __name__ == '__main__':
    _ = today_position()