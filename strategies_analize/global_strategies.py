import pandas as pd
import numpy as np
import pandas_ta as ta
import sys
from scipy.stats import norm
sys.path.append("..")


def rsidi_counter(df_raw, slow, fast, symbol):
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


def macdd_counter(df_raw, slow, fast, symbol):
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


def avs1s_trend(df_raw, slow, fast, symbol):
    slow += 5
    fast += 5
    df = df_raw.copy()
    df['vol_sum'] = df.volume.rolling(slow).sum()
    df['atr'] = df.ta.atr(length=slow) * df['vol_sum']
    df['my_atr'] = (df.high-df.low).rolling(slow).sum() * df['vol_sum']
    df['atr_up'] = df['atr'] + df['atr'].rolling(round(fast)).std()
    df['atr_down'] = df['atr'] - df['atr'].rolling(round(fast)).std()
    df['atr_norm'] = ((df['atr'] - df['atr_down'].rolling(fast).min())/ \
        (df['atr_up'].rolling(fast).max() - df['atr_down'].rolling(fast).min()))
    df['atr_norm'] = df['atr_norm'] - 0.5
    df['atr_norm_ma'] = df['atr_norm'].rolling(fast).mean()
    k = df.ta.stoch(k=fast).iloc[:,0]
    d = df.ta.stoch(k=fast, d=slow).iloc[:,1]
    df['k'] = k * df['atr_norm']
    df['d'] = d * df['atr_norm']
    df['stance'] = np.where((df['k']<0)&(df['k'].shift(slow)>0)&(df['k']<df['d']), -1, np.NaN)
    df['stance'] = np.where((df['k']>0)&(df['k'].shift(slow)<0)&(df['k']>df['d']), 1, df.stance)
    df['stance'] = df['stance'].ffill()
    position = df['stance'].iloc[-1]
    return df, position


def dnert_counter(df_raw, slow, fast, symbol):
    df = df_raw.copy()
    df['Trend_High'] = df['high'].rolling(slow).max()
    df['Trend_Low'] = df['low'].rolling(slow).min()
    df['Fast_Osc'] = df.ta.trima(length=fast)
    df['Long_Entry'] = (df['low'] < df['Trend_Low'].shift()) & (df['Fast_Osc'] > df['close'])
    df['Short_Entry'] = (df['high'] > df['Trend_High'].shift()) & (df['Fast_Osc'] < df['close'])
    df['stance'] = np.nan
    df.loc[df['Long_Entry'], 'stance'] = 1
    df.loc[df['Short_Entry'], 'stance'] = -1
    df['stance'] = df['stance'].ffill()
    position = df['stance'].iloc[-1]
    return df, position


def atr11_counter(df_raw, atr_period, atr_mult, symbol):
    df = df_raw.copy()
    df['ATR'] = df['high'].rolling(atr_period).max() - df['low'].rolling(atr_period).min()
    df['Upper_Band'] = df['close'].rolling(atr_period).mean() + atr_mult/50 * df['ATR']
    df['Lower_Band'] = df['close'].rolling(atr_period).mean() - atr_mult/50 * df['ATR']
    df['Long_Entry'] = (df['high'] > df['Upper_Band'].shift(1))#&(df['volume']>df['volume'].shift(1))
    df['Short_Entry'] = (df['low'] < df['Lower_Band'].shift(1))#&(df['volume']>df['volume'].shift(1))
    df['stance'] = np.nan
    df.loc[df['Long_Entry'], 'stance'] = -1
    df.loc[df['Short_Entry'], 'stance'] = 1
    df['stance'] = df['stance'].ffill()
    position = df['stance'].iloc[-1]
    return df, position


def eng2m_counter(df_raw, slow, fast, symbol):
    df = df_raw.copy()
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
    position = df['stance'].iloc[-1]
    return df, position


def hhllx_counter(df_raw, slow, fast, symbol):
    df = df_raw.copy()
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
    position = df['stance'].iloc[-1]
    return df, position


def ema1b_trend(df_raw, slow, fast, symbol):
    df = df_raw.copy()
    fast = fast-1
    df['ema'] = df.ta.ema(length=slow)
    df['ema_high'] = ta.sma(df.high, length=slow*5)
    df['ema_low'] = ta.sma(df.low, length=slow*5)
    df['max_']=df['high'].rolling(slow).max()
    df['min_']=df['low'].rolling(slow).min()
    df['new_max'] = np.where((df['max_'] > df['max_'].shift(fast))&(df['close']>df['ema_high']), 1, np.NaN)
    df['new_max'] = np.where(df['max_'] < df['max_'].shift(fast), 0, df['new_max'])
    df['new_max'] = df['new_max'].ffill()
    df['new_min'] = np.where((df['min_'] < df['min_'].shift(fast))&(df['close']<df['ema_low']), 1, np.NaN)
    df['new_min'] = np.where(df['min_'] > df['min_'].shift(fast), 0, df['new_min'])
    df['new_min'] = df['new_min'].ffill()
    df['stance'] = np.where((df['new_max'] == 1)&(df['new_min']==0), -1, np.NaN)
    df['stance'] = np.where((df['new_max'] == 0)&(df['new_min']==1), 1, df['stance'])
    df['stance'] = df['stance'].ffill()
    position = df['stance'].iloc[-1]
    return df, position


# def sup1n_trend(df_raw, slow, fast, symbol):
#     df = df_raw.copy()
#     # import in function for report script
#     from app.bot_functions import find_support_resistance_numpy
#     df = find_support_resistance_numpy(df, slow, fast)
#     df['stance'] = np.where(df['close'] < df['support'], -1, np.NaN)
#     df['stance'] = np.where(df['close'] > df['resistance'], 1, df['stance'])
#     df['stance'] = df['stance'].ffill()
#     position = df['stance'].iloc[-1]
#     return df, position


def zsnew_trend(df_raw, slow, fast, symbol):
    df = df_raw.copy()
    x = fast/20
    df['rolling_mean'] = df['close'].rolling(window=slow).mean()
    df['rolling_std'] = df['close'].rolling(window=slow).std()
    df['up'] = df['rolling_mean'] + (0.5+x)*df['rolling_std']
    df['down'] = df['rolling_mean'] - (0.5+x)*df['rolling_std']
    df['rolling_zscore'] = (df['close'] - df['rolling_mean']) / df['rolling_std']
    df['rolling_zscore_mean'] = df['rolling_zscore'].rolling(slow).mean()
    df['rolling_zscore_std']= df['rolling_zscore'].rolling(slow).std()
    df['rolling_zscore_mean_up'] = df['rolling_zscore_mean'] + x*df['rolling_zscore_std']
    df['rolling_zscore_mean_down'] = df['rolling_zscore_mean'] - x*df['rolling_zscore_std']
    df['stance'] = np.where((df['rolling_zscore']>df['rolling_zscore_mean'])&
                            (df['close']<df['up'].shift())
                            , 1, np.nan)
    df['stance'] = np.where((df['rolling_zscore']<df['rolling_zscore_mean'])&
                            (df['close']>df['down'].shift())
                            , -1, df['stance'])
    df['stance'] = np.where((df['rolling_zscore']>df['rolling_zscore_mean_up'])&
                            (df['close']<df['up'].shift())&
                            (df['close']>df['down'].shift())
                            , -1, df['stance'])
    df['stance'] = np.where((df['rolling_zscore']<df['rolling_zscore_mean_down'])&
                            (df['close']>df['down'].shift())&
                            (df['close']<df['up'].shift())
                            , 1, df['stance'])
    df['stance'] = df['stance'].ffill()
    position = df['stance'].iloc[-1]
    return df, position


def mo1to_trend(df_raw, long, short, symbol):
    df = df_raw.copy()
    df['ma1'] = df.ta.t3(length=long, a=0.7)
    df['ma2'] = df.ta.t3(length=short, a=0.7)
    df['stance'] = np.where((df['close']> df['ma1'])&(df['ma1']> df['ma2']), 1, np.nan)
    df['stance'] = np.where((df['close']< df['ma1'])&(df['ma1']< df['ma2']), -1, df['stance'])
    df['stance'] = df['stance'].ffill()
    position = df['stance'].iloc[-1]
    return df, position


def engulf_counter(df_raw, long, short, symbol):
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
    position = df['stance'].iloc[-1]
    return df, position


def rsitr_trend(df_raw, slow, fast, symbol):
    df = df_raw.copy()
    df['ma'] = df.ta.sma(length=slow)
    df['rsi'] = df.ta.rsi(length=fast)
    df['close_std'] = df['close'].rolling(slow).std()
    df['bollup'] = df['ma'] + (fast/10)*df['close_std']
    df['bolldown'] = df['ma'] - (fast/10)*df['close_std']
    df['stance'] = np.where((df['low'].shift(3) < df['bolldown'].shift(3))&
                            (df['low'].shift(2) > df['bolldown'].shift(2))&
                            (df['low'].shift(1) > df['bolldown'].shift(1))&
                            (df['rsi']>df['rsi'].shift(1))
                            , 1, np.NaN)
    df['stance'] = np.where((df['high'].shift(3) > df['bollup'].shift(3))&
                            (df['high'].shift(2) < df['bollup'].shift(2))&
                            (df['high'].shift(1) < df['bollup'].shift(1))&
                            (df['rsi']<df['rsi'].shift(1))
                            , -1, df['stance'])
    df['stance'] = df['stance'].ffill()
    position = df['stance'].iloc[-1]
    return df, position


def madif_trend(df_raw, slow, fast, symbol):

    def cmma_series(df_raw, fast, slow):
        df = df_raw.copy()
        df['close_log'] = np.log(df['close'])
        df['high_log'] = np.log(df['high'])
        df['low_log'] = np.log(df['low'])
        df['mean_log'] = df['close_log'].rolling(fast).mean().shift()
        df['atr'] = df.ta.atr(length=slow)
        fast_sqr = (fast+1)**0.5
        df['indicator'] = 100*norm.cdf(((df['close_log']+df['high_log']+df['low_log'])/3-df['mean_log'])/(df['atr']*fast_sqr))-50
        df['indicator'] = 0.5*df['indicator'].shift(2) + 0.35*df['indicator'].shift() + 0.15*df['indicator']
        return df['indicator']

    df = df_raw.copy()
    df['adj'] = (df['high'] + df['close'] + df['low'])/3
    df['corr1'] = 100*norm.cdf(df.close.rolling(slow).corr(df.high)/((slow)-1)**0.5)-50
    df['corr2'] = 100*norm.cdf(df.close.rolling(slow).corr(df.low)/((slow)-1)**0.5)-50
    df['corr'] = df['corr1'] - df['corr2']
    df['cma'] = df['corr'].rolling(int(fast*2)).mean()
    df['indicator1'] = cmma_series(df, fast, slow)
    df['stance'] = np.where((df['corr'] > df['cma'])&(df['indicator1']<0), 1, np.nan)
    df['stance'] = np.where((df['corr'] < df['cma'])&(df['indicator1']>0), -1, df['stance'])
    df['stance'] = df['stance'].ffill()
    position = df['stance'].iloc[-1]
    return df, position