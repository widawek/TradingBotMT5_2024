import pandas as pd
import numpy as np
import sys
import math
from scipy.stats import linregress
sys.path.append("..")


def exponential_penalty(dfx, alpha: int = 10) -> float:
    cross = np.where(dfx.stance != dfx.stance.shift(), 1, 0)
    density = cross.sum()/len(dfx)
    threshold = 0.055  # Próg akceptowalnej gęstości
    return math.exp(-alpha * max(0, density - threshold))


def sharpe_ratio(returns, risk_free_rate=0):
    """Sharpe ratio = (średni zwrot - stopa wolna od ryzyka) / odchylenie standardowe zwrotów"""
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()


def omega_ratio(returns, threshold=0):
    """Omega ratio = suma zwrotów powyżej threshold / suma zwrotów poniżej threshold"""
    gains = returns[returns > threshold].sum()
    losses = abs(returns[returns < threshold].sum())
    return gains / losses if losses != 0 else 1.2


def max_drawdown(strategy):
    """Max Drawdown = największe obsunięcie kapitału"""
    peak = strategy.cummax()
    drawdown = (strategy - peak) / peak
    return drawdown.min()/100


def ulcer_index(strategy):
    """Ulcer Index = średnia kwadratowa drawdownów"""
    peak = strategy.cummax()
    drawdown = (strategy - peak) / peak
    return np.sqrt(np.mean(drawdown**2))


def cagr(strategy, years):
    """CAGR = skumulowana roczna stopa zwrotu"""
    return (strategy.iloc[-1] / strategy.iloc[0])**(1/years) - 1


def equity_curve_stability(strategy):
    """R^2 dopasowania equity curve do regresji liniowej"""
    x = np.arange(len(strategy))
    log_strategy = np.log(strategy)
    slope, intercept, r_value, _, _ = linregress(x, log_strategy)
    return r_value**2


def complex(returns, sharpe_multiplier, years=1):
    """Finalna metryka"""
    years=1
    sharpe = sharpe_multiplier*sharpe_ratio(returns)
    omega = omega_ratio(returns)
    strategy = (1+returns).cumprod()
    #mdd = abs(max_drawdown(strategy))
    ui = ulcer_index(strategy)
    cagr_value = cagr(strategy, years)
    stability = equity_curve_stability(strategy)
    return (sharpe * omega * cagr_value * stability) / ui


def complex_metric(dfx, penalty=True):
    df = dfx.dropna().copy()
    penalty_ = exponential_penalty(df) if penalty else 1
    return round(complex(df['returns'], 1, years=1)*penalty_, 6)


def wlr_rr_metric(dfx, penalty=True):
    df = dfx.dropna().copy()
    reward = df[df['return']>0]
    risk = df[df['return']<0]
    risk_reward_ratio = round(reward['return'].mean() / risk['return'].mean(), 5)
    win_loss_ratio = round(len(reward)/len(risk), 5)
    penalty_ = exponential_penalty(df) if penalty else 1
    return round(risk_reward_ratio*win_loss_ratio*penalty_, 6)


def mix_rrsimple_metric(dfx, penalty=True):
    df = dfx.dropna().copy()
    reward = df[df['return']>0]
    risk = df[df['return']<0]
    risk_reward_ratio = round(reward['return'].mean() / risk['return'].mean(), 3)
    win_loss_ratio = round(len(reward)/len(risk), 5)
    df['strategy'] = (1+df['return']).cumprod() - 1
    penalty_ = exponential_penalty(df) if penalty else 1
    return round(np.mean([df['strategy'].min(), df['strategy'].max(), df['strategy'].iloc[-1]])*risk_reward_ratio*win_loss_ratio*penalty_, 6)


def only_strategy_metric(dfx, penalty=True):
    df = dfx.dropna().copy()
    df['strategy'] = (1+df['return']).cumprod() - 1
    penalty_ = exponential_penalty(df) if penalty else 1
    return round(np.mean([df['strategy'].min(), df['strategy'].max(), df['strategy'].iloc[-1]])*penalty_, 6)


def min_final_strategy_metric(dfx, penalty=True):
    df = dfx.dropna().copy()
    df['strategy'] = (1+df['return']).cumprod() - 1
    penalty_ = exponential_penalty(df) if penalty else 1
    return round(np.mean([df['strategy'].min(), df['strategy'].iloc[-1]])*penalty_, 6)


def sharpe_metric(dfx, penalty=True):
    df = dfx.dropna().copy()
    penalty_ = exponential_penalty(df) if penalty else 1
    return round((df['return'].mean()/df['return'].std())*penalty_, 6)


def sharpe_drawdown_metric(dfx, penalty=True):
    df = dfx.dropna().copy()
    strategy = (1+df['return']).cumprod()
    penalty_ = exponential_penalty(df) if penalty else 1
    return round((df['return'].mean()/df['return'].std())*(1+max_drawdown(strategy))*penalty_, 6)


def sharpe_drawdown_daily_metric(dfx, penalty=True):
    df = dfx.dropna().copy()
    df['date'] = df['time'].dt.date
    daily_returns = df.groupby('date')['return'].apply(lambda r: (1 + r).prod() - 1)
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    strategy_ = (1 + daily_returns).cumprod() - 1
    drawdown = max_drawdown(strategy_)
    penalty_ = exponential_penalty(df) if penalty else 1
    return round(sharpe * (1 + drawdown) * penalty_, 6)
