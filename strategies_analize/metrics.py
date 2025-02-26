import pandas as pd
import numpy as np


def wlr_rr_metric(dfx):
    df = dfx.copy()
    df = df.dropna()
    reward = df[df['return']>0]
    risk = df[df['return']<0]
    risk_reward_ratio = round(reward['return'].mean() / risk['return'].mean(), 3)
    win_loss_ratio = round(len(reward)/len(risk), 5)
    return round(risk_reward_ratio*win_loss_ratio, 8)


def mix_rrsimple_metric(dfx):
    df = dfx.copy()
    df = df.dropna()
    reward = df[df['return']>0]
    risk = df[df['return']<0]
    risk_reward_ratio = round(reward['return'].mean() / risk['return'].mean(), 3)
    win_loss_ratio = round(len(reward)/len(risk), 5)
    df['strategy'] = (1+df['return']).cumprod() - 1
    return round(np.mean([df['strategy'].min(), df['strategy'].max(), df['strategy'].iloc[-1]])*risk_reward_ratio*win_loss_ratio, 8)


def only_strategy_metric(dfx):
    df = dfx.copy()
    df = df.dropna()
    df['strategy'] = (1+df['return']).cumprod() - 1
    return round(np.mean([df['strategy'].min(), df['strategy'].max(), df['strategy'].iloc[-1]]), 5)


def sharpe_metric(dfx):
    df = dfx.copy()
    df = df.dropna()
    return round(df['return'].mean()/df['return'].std(), 5)