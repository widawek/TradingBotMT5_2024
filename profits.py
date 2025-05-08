import MetaTrader5 as mt
import pandas as pd
import numpy as np
from datetime import datetime as dt
from datetime import timedelta
from time import sleep
from app.functions import *
from config.parameters import symbols
import matplotlib.pyplot as plt
mt.initialize()


def correlations(symbols):
    symbols = [symbol for symbol in symbols if 'USD' in symbol]

    def returns_symbol(symbol):
        df = get_data(symbol, 'M1', 1, 500)
        df['returns'] = np.log(df.close/df.close.shift())
        if not symbol.startswith('USD'):
            df['returns'] = df['returns'] * -1
        return df['returns'].tolist()
    retrns = {}
    for symbol in symbols:
        retrns[symbol] = returns_symbol(symbol)

    df = pd.DataFrame(retrns)
    corr_matrix = df.corr()
    det_corr = np.linalg.det(corr_matrix)
    print(f"Wyznacznik macierzy korelacji: {det_corr:.2f}")
    return det_corr


def profit_chart():
    results = []
    i = 0
    start_balance = mt.account_info().balance - closed_pos()
    while True:
        info = mt.account_info()
        balance = info.balance
        free_margin =  info.margin
        mrgintobalance = free_margin/balance
        correlation = correlations(symbols)
        profit = sum([i.profit for i in mt.positions_get()])
        percent = round((((balance-start_balance)+profit)/start_balance)*100, 2)
        results.append((dt.now(), start_balance, balance, profit, percent,
                        mrgintobalance, correlation))
        max_percent = max([i[4] for i in results])
        min_percent = min([i[4] for i in results])
        print("Max: ", max_percent)
        print("Min: ", min_percent)
        sleep(5)
        i += 1
        try:
            if i % 50 == 0:
                df = pd.DataFrame(results, columns=[
                    'time', 'start_balance', 'balance', 'positions_profit',
                    'percent', 'mrgintobalance', 'correlation'])
                #df.to_excel('test.xlsx')
                df['correlation'] = df['correlation'].rolling(40).mean()
                df['zero'] = 0
                df.set_index('time', inplace=True)
                fig = plt.figure(figsize=(12, 8))
                ax1 = plt.subplot(311)
                plt.plot(df['percent'], c='b', label="Percentage")
                #plt.plot(df.zero)
                ax2 = plt.subplot(312, sharex=ax1)
                plt.plot(df.mrgintobalance, c='r', label="Margin to balance")
                ax3 = plt.subplot(313, sharex=ax1)
                plt.plot(df.correlation, c='y', label="Symbols correlation to USD")
                date = dt.now().date().strftime('%d-%m-%Y')
                plt.savefig(f'charts\\rezultat_{date}.png', dpi=300, bbox_inches='tight')
                _ = plt.close()

        except Exception as e:
            print(e)
            pass


