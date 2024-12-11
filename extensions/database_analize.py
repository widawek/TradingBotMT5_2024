from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
import pandas as pd
import MetaTrader5 as mt
from datetime import datetime as dt
from datetime import timedelta
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Utwórz bazę do definicji modeli
from app.database_class import ReadDatabase
Base = declarative_base()


def report_(by_, from_, to_, excel=False):
    from extensions.result_report import get_raw_close_positions_data
    from numpy import vectorize
    import matplotlib.pyplot as plt
    import numpy as np
    from app.functions import pandas_options
    # Przykład użycia
    db_manager = ReadDatabase()

    # Odczyt danych z tabeli 'Position' do pandas DataFrame
    positions_profits, _ = get_raw_close_positions_data(from_, to_)
    def give_me_profit(ticket):
        try:
            df = positions_profits[positions_profits['position_id']==ticket]
        except TypeError:
            return 0.0
        try:
            return df['profit'].iloc[-1]
        except IndexError:
            return 0.0

    vectorize_ = vectorize(give_me_profit)
    df_positions = db_manager.read_positions_to_df()
    df_positions['close_profit'] = vectorize_(df_positions['ticket'])
    print(df_positions.close_profit.sum())
    #Odczyt danych z tabeli 'Profit' do pandas DataFrame
    df_profits = db_manager.read_profits_to_df()
    #df_profits = df_profits[df_profits['volume_condition']==False]

    def group_profit_by(what, to_excel=False):
        print(f'\n{what}')
        grouped_profit_max = df_profits.groupby('ticket').agg({'profit': 'max'})
        grouped_profit_min = df_profits.groupby('ticket').agg({'profit': 'min'})
        grouped_profit_mean = df_profits.groupby('ticket').agg({'profit': 'mean'})
        grouped_profit_counter_max = df_profits.groupby('ticket').agg({'fake_position_counter': 'max'})
        grouped_profit_counter_mean = df_profits.groupby('ticket').agg({'fake_position_counter': 'mean'})
        grouped_profit_max.reset_index(inplace=True)
        grouped_profit_min.reset_index(inplace=True)
        grouped_profit_mean.reset_index(inplace=True)
        grouped_profit_counter_max.reset_index(inplace=True)
        grouped_profit_counter_mean.reset_index(inplace=True)
        df_positions['max_profit'] = grouped_profit_max['profit'] / df_positions['calculated_profit']
        df_positions['min_profit'] = grouped_profit_min['profit'] / df_positions['calculated_profit']
        df_positions['mean_profit'] = grouped_profit_mean['profit'] / df_positions['calculated_profit']
        df_positions['max_counter'] = grouped_profit_counter_max['fake_position_counter']
        df_positions['mean_counter'] = grouped_profit_counter_mean['fake_position_counter']
        a = df_positions.groupby(what).agg(max_profit=('max_profit', 'mean'),
                                        std_max=('max_profit', 'std'),
                                        std_min=('min_profit', 'std'))
        b = df_positions.groupby(what).agg(min_profit=('min_profit', 'mean'))
        c = df_positions.groupby(what).agg(mean_profit=('mean_profit', 'mean'),
                                        counter=('mean_profit', 'count'),
                                        close_profit=('close_profit', 'sum'),
                                        fake_position_counter=('max_counter', 'max'),
                                        fake_position_mean=('mean_counter', 'mean'))
        a['min_profit'] = b['min_profit']
        a['mean_profit'] = c['mean_profit']
        a['max-min'] = a['max_profit'] + a['min_profit']
        a['close_profit'] = c['close_profit']
        a['counter'] = c['counter']
        a['pos_counter'] = c['fake_position_counter']
        a['pos_mean'] = c['fake_position_mean']
        a = a[a['pos_counter'] > 0]
        if to_excel:
            a.to_excel("results.xlsx")
        print(a)

    def plot_profits(df, symbol, days_to_past, condition=""):
        df['time'] = pd.to_datetime(df['time'])
        df = df[df['time'].dt.date >= dt.now().date() - timedelta(days=days_to_past)]
        groups = df.groupby('ticket')
        print(groups)
        #df_positions = df_positions[df_positions['ticket'].isin()]
        if symbol == 'all':
            tickets=df_positions['ticket'].to_list()
            calc_profs=df_positions['calculated_profit'].to_list()
            #close_profits=df_positions['close_profit'].to_list()
        else:
            tickets=df_positions[df_positions['symbol']==symbol]['ticket'].to_list()
            calc_profs=df_positions[df_positions['symbol']==symbol]['calculated_profit'].to_list()
            #close_profits=df_positions[df_positions['symbol']==symbol]['close_profit'].to_list()
        # Tworzenie wykresu
        plt.figure(figsize=(30, 30))
        i = 0
        spreads = []
        lists = []
        for name, group in groups:
            #by_what = group['mean_profit']
            by_what = calc_profs[i]
            #by_what = group['profit_max'].iloc[-1]
            what = group['profit']/by_what
            what1 = group['profit'].iloc[-1]/by_what
            if name in tickets:
                if condition=='>':
                    if group['profit'].iloc[-1] > 0:
                        line, = plt.plot(range(len(group['profit'])), what, label=name, linewidth=0.5)
                        plt.plot(len(group['profit'])-1, what1, marker='*', color=line.get_color(), markersize=10)
                        spreads.append(group['profit'].iloc[0])
                        x = group['profit']/by_what
                        lists.append(x.to_list())
                    i+=1
                elif condition=='<':
                    if group['profit'].iloc[-1] < 0:
                        line, = plt.plot(range(len(group['profit'])), what, label=name, linewidth=0.5)
                        plt.plot(len(group['profit'])-1, what1, marker='*', color=line.get_color(), markersize=10)
                        spreads.append(group['profit'].iloc[0])
                        x = group['profit']/by_what
                        lists.append(x.to_list())
                    i+=1
                else:
                    line, = plt.plot(range(len(group['profit'])), what, label=name, linewidth=0.5)
                    plt.plot(len(group['profit'])-1, what1, marker='*', color=line.get_color(), markersize=10)
                    #x = group['profit']/by_what
                    lists.append(what.to_list())
                    i+=1
                    spreads.append(group['profit'].iloc[0])

        max_len = max(len(sublist) for sublist in lists)
        # Tworzymy nową listę, w której każdy indeks to średnia z odpowiednich indeksów
        averages = []
        for i in range(max_len):
            # Zbieramy wszystkie wartości z danego indeksu, uwzględniając tylko istniejące elementy
            values = [sublist[i] for sublist in lists if i < len(sublist)]
            # Obliczamy średnią tylko dla tych wartości, które są dostępne
            avg = np.mean(values)
            # Dodajemy do listy wynikowej
            averages.append(avg)
        plt.plot(range(max_len), averages, linewidth=3, c='black')
        plt.title(f"Spread on {symbol}: {round(np.mean(spreads), 2)} $")
        plt.show()
    pandas_options()
    group_profit_by(by_, False)
    #group_profit_by('tiktok')
    plot_profits(df_profits, 'all', 0, condition='')
report_(['tiktok'], 0, -2)
