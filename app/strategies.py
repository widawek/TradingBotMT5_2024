import pandas_ta as ta
import numpy as np
import pandas as pd


def technique3(df_raw, slow, fast):
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


def rsi_divergence_strategy(df_raw, slow, fast):
    df = df_raw.copy()
    df['rsi'] = ta.rsi(df['close'], length=slow)
    df['price_peak'] = df['high'].rolling(fast).max()
    df['price_trough'] = df['low'].rolling(fast).min()
    df['rsi_peak'] = df['rsi'].rolling(fast).max()
    df['rsi_trough'] = df['rsi'].rolling(fast).min()

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


def moving_averages(df_raw, slow, fast):
    df = df_raw.copy()
    df['adj'] = (df['close'] + df['high'] + df['low']) / 3
    ma1 = df.ta.vwma(length=fast)
    ma2 = ta.vwma(df['adj'], df['volume'], length=slow)
    df['stance'] = np.where(ma1>=ma2, 1, 0)
    df['stance'] = np.where(ma1<ma2, -1, df['stance'])
    position = df['stance'].iloc[-1]
    return df, position
