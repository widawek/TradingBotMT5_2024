import pandas as pd
import numpy as np
import pandas_ta as ta


def find_support_resistance_numpy(df, slow, fast):
    df['tolerance'] = (df.high-df.low).rolling(slow).mean() * fast/20
    low = df['low'].to_numpy()
    high = df['high'].to_numpy()
    tolerance = df['tolerance'].to_numpy()

    # Znajdź lokalne minima i maksima za pomocą NumPy
    is_support = np.where((low[1:-1] < low[:-2] - tolerance[1:-1]) & (low[1:-1] < low[2:] - tolerance[1:-1]))[0] + 1
    is_resistance = np.where((high[1:-1] > high[:-2] + tolerance[1:-1]) & (high[1:-1] > high[2:] + tolerance[1:-1]))[0] + 1

    # Utwórz kolumny support i resistance
    df['support'] = np.nan
    df['resistance'] = np.nan
    df.loc[df.index[is_support], 'support'] = df['low'].iloc[is_support]
    df.loc[df.index[is_resistance], 'resistance'] = df['high'].iloc[is_resistance]

    # Wypełnij brakujące wartości NaN poprzednimi wartościami (forward fill)
    df['support'] = df['support'].ffill()
    df['resistance'] = df['resistance'].ffill()
    return df


def mas_t3_mad(df_raw, long, short):
    df = df_raw.copy()
    df['ma1'] = df.ta.t3(length=long, a=0.7)
    df['ma2'] = df.ta.t3(length=short, a=0.7)
    df['stance'] = np.where((df['close']> df['ma1'])&(df['ma1']> df['ma2']), 1, np.nan)
    df['stance'] = np.where((df['close']< df['ma1'])&(df['ma1']< df['ma2']), -1, df['stance'])
    df['stance'] = df['stance'].ffill()
    return df


def engulf(df_raw, long, short):
    df = df_raw.copy()
    df['volume_mean'] = df['volume'].rolling(short).mean()
    df['vol_cond'] = df['volume'] > df['volume_mean']
    df['highmean'] = df['high'].rolling(long).mean()
    df['lowmean'] = df['low'].rolling(long).mean()
    df['engulf_long'] = np.where((df['close'].shift()<df['open'].shift())&(df['close']>df['open'])&(df['high']>df['high'].shift())&(df['close']>df['open'].shift()), 1, 0)
    df['engulf_short'] = np.where((df['close'].shift()>df['open'].shift())&(df['close']<df['open'])&(df['low']<df['low'].shift())&(df['close']<df['open'].shift()), 1, 0)
    df['engulf_sum'] = df['engulf_long'].rolling(long).sum() - df['engulf_short'].rolling(long).sum()
    df['stance'] = np.where((df['engulf_sum'] > 0)&(df['vol_cond'])&(df['close']>df['lowmean']), -1, np.NaN)
    df['stance'] = np.where((df['engulf_sum'] < 0)&(df['vol_cond'])&(df['close']<df['highmean']), 1, df['stance'])
    df['stance'] = df['stance'].ffill()
    return df


def rsi_divergence_strategy(df_raw, slow, fast):
    df = df_raw.copy()
    df['rsi'] = ta.rsi(df['close'], length=slow)
    df['price_peak'] = df['high'].rolling(fast).max()
    df['price_trough'] = df['low'].rolling(fast).min()
    df['rsi_peak'] = df['rsi'].rolling(fast).max()
    df['rsi_trough'] = df['rsi'].rolling(fast).min()

    df['bullish_div'] = np.where(
        ((df['price_trough'] < df['price_trough'].shift(1)) &  # Price makes a lower low
        (df['rsi_trough'] > df['rsi_trough'].shift(1))),  # RSI makes a higher low
        1, 0)

    df['bearish_div'] = np.where(
        ((df['price_peak'] > df['price_peak'].shift(1)) &  # Price makes a higher high
        (df['rsi_peak'] < df['rsi_peak'].shift(1))),  # RSI makes a lower high
        1, 0)

    df['stance'] = np.NaN
    df.loc[df['bullish_div'] == 1, 'stance'] = 1  # Buy signal on bullish divergence
    df.loc[df['bearish_div'] == 1, 'stance'] = -1  # Sell signal on bearish divergence
    df['stance'] = df['stance'].ffill()
    return df


def t52_moving_average_close(df_raw, slow, fast):
    df = df_raw.copy()
    df['adj'] = (df['close'] + df['high'] + df['low']) / 3
    ma1 = ta.t3(df['adj'], length=round(fast*slow/4), a=0.6)
    df['stance'] = np.where((df['close']>ma1), 1, np.NaN)
    df['stance'] = np.where((df['close']<ma1), -1, df['stance'])
    df['stance'] = df['stance'].ffill()
    return df


def macd_divergence_strategy(df_raw, slow, fast):
    df = df_raw.copy()
    macd = df.ta.macd(fast=round(slow), slow=round(slow*2), signal=round(slow*3/4))
    df['rsi'] = macd.iloc[:,0]
    df['price_peak'] = df['high'].rolling(fast).max()
    df['price_trough'] = df['low'].rolling(fast).min()
    df['rsi_peak'] = df['rsi'].rolling(fast).max()
    df['rsi_trough'] = df['rsi'].rolling(fast).min()

    df['bullish_div'] = np.where(
        ((df['price_trough'] < df['price_trough'].shift(1)) &  # Price makes a lower low
        (df['rsi_trough'] > df['rsi_trough'].shift(1))),  # RSI makes a higher low
        1, 0)

    df['bearish_div'] = np.where(
        ((df['price_peak'] > df['price_peak'].shift(1)) &  # Price makes a higher high
        (df['rsi_peak'] < df['rsi_peak'].shift(1))),  # RSI makes a lower high
        1, 0)

    df['stance'] = np.NaN
    df.loc[df['bullish_div'] == 1, 'stance'] = 1  # Buy signal on bullish divergence
    df.loc[df['bearish_div'] == 1, 'stance'] = -1  # Sell signal on bearish divergence
    df['stance'] = df['stance'].ffill()
    return df


def avs_aka_atr_vol_stoch(df_raw, slow, fast):
    df = df_raw.copy()
    factor = slow*(fast-1)
    df['vol_sum'] = df.volume.rolling(factor).sum()
    df['atr'] = df.ta.atr(length=factor) * df['vol_sum']
    df['my_atr'] = (df.high-df.low).rolling(factor).sum() * df['vol_sum']
    df['atr_up'] = df['atr'] + df['atr'].rolling(round(factor)).std()
    df['atr_down'] = df['atr'] - df['atr'].rolling(round(factor)).std()
    df['atr_norm'] = ((df['atr'] - df['atr_down'].rolling(factor).min())/ \
        (df['atr_up'].rolling(factor).max() - df['atr_down'].rolling(factor).min()))
    df['atr_norm'] = df['atr_norm'] - 0.5
    df['atr_norm_ma'] = df['atr_norm'].rolling(factor).mean()
    k = df.ta.stoch(k=factor).iloc[:,0]
    d = df.ta.stoch(k=factor, d=factor).iloc[:,1]
    df['k'] = k * df['atr_norm']
    df['d'] = d * df['atr_norm']
    df['stance'] = np.where((df['k']<0)&(df['k'].shift(factor)>0)&(df['k']<df['d']), -1, np.NaN)
    df['stance'] = np.where((df['k']>0)&(df['k'].shift(factor)<0)&(df['k']>df['d']), 1, df.stance)
    df['stance'] = df['stance'].ffill()
    return df


def engminmax(df, slow, fast):
    fast = fast-1
    df['local_max'] = df.high.rolling(slow).max()
    df['new_max'] = np.where(df['local_max']>df['local_max'].shift(1), 1, 0)
    df['local_min'] = df.low.rolling(slow).min()
    df['new_min'] = np.where(df['local_min']<df['local_min'].shift(1), 1, 0)
    cond_long = (df['close'] > df['close'].shift(1))&(df['close'].shift(1) < df['close'].shift(2))&(df['close'].shift(2) < df['close'].shift(3))&(df['new_min'].rolling(fast).sum()>=1)#&(df['close'] > df['close'].shift(2))
    cond_short = (df['close'] < df['close'].shift(1))&(df['close'].shift(1) > df['close'].shift(2))&(df['close'].shift(2) > df['close'].shift(3))&(df['new_max'].rolling(fast).sum()>=1)#&(df['close'] < df['close'].shift(2))
    df['stance'] = np.where(cond_long, 1, np.NaN)
    df['stance'] = np.where(cond_short, -1, df['stance'])
    df['stance'] = df['stance'].ffill()
    return df


def hhll(df, slow, fast):
    fast = fast-1
    df['max_']=df['high'].rolling(slow).max()
    df['min_']=df['low'].rolling(slow).min()
    df['max_low'] = np.where(df['max_'] > df['max_'].shift(1), df['low'].rolling(fast).min(), np.NaN)
    df['max_low'] = df['max_low'].ffill()
    df['min_high'] = np.where(df['min_'] < df['min_'].shift(1), df['high'].rolling(fast).max(), np.NaN)
    df['min_high'] = df['min_high'].ffill()
    df['new_max'] = np.where(df['max_'] > df['max_'].shift(1), 1, np.NaN)
    df['new_max'] = np.where(df['max_'] < df['max_'].shift(1), 0, df['new_max'])
    df['new_max'] = df['new_max'].ffill()
    df['new_min'] = np.where(df['min_'] < df['min_'].shift(1), 1, np.NaN)
    df['new_min'] = np.where(df['min_'] > df['min_'].shift(1), 0, df['new_min'])
    df['new_min'] = df['new_min'].ffill()
    df['stance'] = np.where((df['new_max'] == 1)&(df['new_min']==0)&(df['close']>=df['min_'])&(df['close']<=df['min_high']), 1, np.NaN)
    df['stance'] = np.where((df['new_max'] == 1)&(df['new_min']==0)&(df['close']>df['max_'].shift(1)), -1, df['stance'])
    df['stance'] = np.where((df['new_max'] == 0)&(df['new_min']==1)&(df['close']<=df['max_'])&(df['close']>=df['max_low']), -1, df['stance'])
    df['stance'] = np.where((df['new_max'] == 0)&(df['new_min']==1)&(df['close']<df['min_'].shift(1)), 1, df['stance'])
    df['stance'] = df['stance'].ffill()
    return df


def emaboll(df, slow, fast):
    fast = fast-1
    df['ema'] = df.ta.ema(length=slow)
    df['ema_high'] = ta.sma(df.high, length=slow*5)
    df['ema_low'] = ta.sma(df.low, length=slow*5)
    df['std'] = df.ta.stdev(length=slow)
    df['upper'] = df['ema'] + (fast/10)*df['std']
    df['lower'] = df['ema'] - (fast/10)*df['std']
    df['max_']=df['high'].rolling(slow).max()
    df['min_']=df['low'].rolling(slow).min()
    df['new_max'] = np.where((df['max_'] > df['max_'].shift(1))&(df['close']>df['ema_high']), 1, np.NaN)
    df['new_max'] = np.where(df['max_'] < df['max_'].shift(1), 0, df['new_max'])
    df['new_max'] = df['new_max'].ffill()
    df['new_min'] = np.where((df['min_'] < df['min_'].shift(1))&(df['close']<df['ema_low']), 1, np.NaN)
    df['new_min'] = np.where(df['min_'] > df['min_'].shift(1), 0, df['new_min'])
    df['new_min'] = df['new_min'].ffill()
    df['stance'] = np.where((df['new_max'] == 1)&(df['new_min']==0), -1, np.NaN)
    df['stance'] = np.where((df['new_max'] == 0)&(df['new_min']==1), 1, df['stance'])
    df['stance'] = df['stance'].ffill()
    return df


def sup_res_numpy(df, slow, fast):
    # import in function for report script
    df = find_support_resistance_numpy(df, slow, fast)
    df['stance'] = np.where(df['close'] < df['support'], -1, np.NaN)
    df['stance'] = np.where(df['close'] > df['resistance'], 1, df['stance'])
    df['stance'] = df['stance'].ffill()
    return df


def altrend(df, long, short):
    long *= 2
    short *= 2
    df['res_long'] = df.close-df.open
    df['dir_long'] = np.where(df.res_long>0, 1, -1)
    df['trend_long'] = df['res_long'].rolling(long).sum()*df['dir_long'].rolling(short).mean()

    df['res_short'] = df.open-df.close
    df['dir_short'] = np.where(df.res_short>0, 1, -1)
    df['trend_short'] = df['res_short'].rolling(long).sum()*df['dir_short'].rolling(short).mean()

    df['stance'] = np.where((df['trend_short']<0)&(df['trend_long']>0), 1, np.NaN)
    df['stance'] = np.where((df['trend_short']>0)&(df['trend_long']<0), -1, df['stance'])
    df['stance'] = df['stance'].ffill()
    return df

