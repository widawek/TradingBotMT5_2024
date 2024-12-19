import pandas_ta as ta
import numpy as np
import pandas as pd

# The first six characters are a mark of strategy, so they should be unique.

# def model_M20():
#     pass


def techn1ique3_trend_M1(df_raw, slow, fast):
    df = df_raw.copy()
    df.set_index(df['time'], inplace=True)
    df['numbers'] = [1+i for i in range(len(df))]
    df['vwap_D'] = df.ta.vwap(anchor="D")
    df['adj'] = (df['close'] + df['high'] + df['low'])/3
    df['ma'] = ta.sma(df['adj'], length=slow)
    df['super_new_indicator'] = (df['close'] - df['close'].shift(1)) * df['volume']
    df['super_new_indicator'] = df['super_new_indicator'].rolling(slow).mean()
    df['super_max'] = df['super_new_indicator'].rolling(slow*fast).max()
    df['super_min'] = df['super_new_indicator'].rolling(slow*fast).min()
    df['median'] = (df['super_max'] + df['super_min']) / 2
    df['super_ma'] = df['ma'] + df['ma']*df['median']
    df['k1'] = df.ta.stoch(k=fast).iloc[:,0]
    df['k2'] = df.ta.stoch(k=slow).iloc[:,0]
    df['stance'] = np.where((df['super_ma'] < df['ma']) & (df.close > df.vwap_D), 1, 0)
    df['stance'] = np.where((df['super_ma'] > df['ma']) & (df.close < df.vwap_D), -1, df['stance'])
    df['stance'] = np.where((df.k1>df.k2)&(df.stance==0), 1, df['stance'])
    df['stance'] = np.where((df.k1<df.k2)&(df.stance==0), -1, df['stance'])
    position = df['stance'].iloc[-1]
    return df, position


def rsi1_divergence_strategy_counter_M1(df_raw, slow, fast):
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
    position = df['stance'].iloc[-1]
    return df, position


def stoch1_divergence_strategy_counter_M1(df_raw, slow, fast):
    df = df_raw.copy()
    df['rsi'] = df.ta.stoch(k=slow).iloc[:,0]
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
    position = df['stance'].iloc[-1]
    return df, position


def cci1_divergence_strategy_counter_M1(df_raw, slow, fast):
    df = df_raw.copy()
    df['rsi'] = df.ta.cci(length=slow)
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
    position = df['stance'].iloc[-1]
    return df, position


def a_moving_averages_trend_M1(df_raw, slow, fast):
    df = df_raw.copy()
    df['adj'] = (df['close'] + df['high'] + df['low']) / 3
    ma1 = df.ta.vwma(length=fast)
    ma2 = ta.vwma(df['adj'], df['volume'], length=slow)
    df['stance'] = np.where(ma1>=ma2, 1, 0)
    df['stance'] = np.where(ma1<ma2, -1, df['stance'])
    position = df['stance'].iloc[-1]
    return df, position


def b_moving_averages_close_trend_M1(df_raw, slow, fast):
    df = df_raw.copy()
    df['adj'] = (df['close'] + df['high'] + df['low']) / 3
    ma1 = df.ta.vwma(length=fast)
    ma2 = ta.vwma(df['adj'], df['volume'], length=slow)
    df['stance'] = np.where(((df['close']>=ma2)&(df['close']>=ma1)), 1, -1)
    position = df['stance'].iloc[-1]
    return df, position


def t3_moving_average_close_trend_M1(df_raw, slow, fast):
    df = df_raw.copy()
    df['adj'] = (df['close'] + df['high'] + df['low']) / 3
    ma1 = ta.t3(df['adj'], length=round(fast*slow/5), a=0.95)
    df['stance'] = np.where((df['close']>ma1), 1, np.NaN)
    df['stance'] = np.where((df['close']<ma1), -1, df['stance'])
    df['stance'] = df['stance'].ffill()
    position = df['stance'].iloc[-1]
    return df, position


def macd1_signal_trend_M1(df_raw, slow, fast):
    df = df_raw.copy()
    macd = df.ta.macd(fast=round(fast), slow=round(slow), signal=round(fast*3/4))
    df['macd'] = macd.iloc[:,0]
    df['signal'] = macd.iloc[:,2]
    df['stance'] = np.where((df['macd']>=df['signal']), 1, -1)
    position = df['stance'].iloc[-1]
    return df, position


def macd2_histogram0_trend_M1(df_raw, slow, fast):
    df = df_raw.copy()
    macd = df.ta.macd(fast=round(fast), slow=round(slow), signal=round(fast*3/4))
    df['histogram'] = macd.iloc[:,1]
    df['stance'] = np.where((df['histogram']>0), 1, -1)
    position = df['stance'].iloc[-1]
    return df, position


def macd3_histogram1_trend_M1(df_raw, slow, fast):
    df = df_raw.copy()
    macd = df.ta.macd(fast=round(fast), slow=round(slow), signal=round(fast*3/4))
    df['histogram'] = macd.iloc[:,1]
    df['stance'] = np.where((df['histogram']>df['histogram'].shift(1)), 1, -1)
    position = df['stance'].iloc[-1]
    return df, position


def stoch2_trend_M1(df_raw, slow, fast):
    df = df_raw.copy()
    df['k1'] = df.ta.stoch(k=fast).iloc[:,0]
    df['k2'] = df.ta.stoch(k=slow).iloc[:,0]
    df['stance'] = np.where(df['k1']>=df['k2'], 1, -1)
    position = df['stance'].iloc[-1]
    return df, position


def macd4_divergence_strategy_counter_M1(df_raw, slow, fast):
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
    position = df['stance'].iloc[-1]
    return df, position


def avs1_aka_atr_vol_stoch_trend_M1(df_raw, slow, fast):
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
    position = df['stance'].iloc[-1]
    return df, position


def momen1tum_divergence_strategy(df, slow, fast):
    df['momentum'] = df['close'] / df.close.rolling(slow).mean()
    df['std'] = df['close'].rolling(slow).std()
    df['lambda'] = df['std'] / df['std'].rolling(slow).mean()
    df['velocity'] = df['momentum'] * df['lambda']
    df['velocity_mean'] = ta.t3(df['velocity'], length=fast)
    df['rsi'] = ta.rsi(df['close'], length=slow)
    df['price_peak'] = df['high'].rolling(fast).max()
    df['price_trough'] = df['low'].rolling(fast).min()
    df['rsi_peak'] = df['velocity_mean'].rolling(fast).max()
    df['rsi_trough'] = df['velocity_mean'].rolling(fast).min()

    df['bullish_div'] = np.where(
        (df['price_trough'] < df['price_trough'].shift(1)) &  # Price makes a lower low
        (df['rsi_trough'] > df['rsi_trough'].shift(1)),  # RSI makes a higher low
        1, 0)

    # Identify bearish divergence
    df['bearish_div'] = np.where(
        (df['price_peak'] > df['price_peak'].shift(1)) &  # Price makes a higher high
        (df['rsi_peak'] < df['rsi_peak'].shift(1)),  # RSI makes a lower high
        1, 0)

    # Generate trading signals
    df['stance'] = np.NaN
    df.loc[df['bullish_div'] == 1, 'stance'] = 1.0  # Buy signal on bullish divergence
    df.loc[df['bearish_div'] == 1, 'stance'] = -1.0  # Sell signal on bearish divergence
    df['stance'] = df['stance'].ffill()
    position = df['stance'].iloc[-1]
    return df, position
