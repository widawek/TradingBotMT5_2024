from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, relationship, declarative_base
import pandas as pd
import MetaTrader5 as mt
from datetime import datetime as dt
from datetime import timedelta

# Utwórz bazę do definicji modeli
Base = declarative_base()

# Model dla tabeli Position
class Position(Base):
    __tablename__ = 'positions'

    ticket = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    pos_type = Column(Integer, nullable=False)
    open_time = Column(Integer, nullable=False)
    volume = Column(Float, nullable=False)
    price_open = Column(Float, nullable=False)
    comment = Column(String, nullable=True)
    reverse_mode = Column(String, nullable=False)
    trigger = Column(String, nullable=False)
    trigger_divider = Column(Float, nullable=False)
    decline_factor = Column(Float, nullable=False)
    profit_factor = Column(Float, nullable=False)
    calculated_profit = Column(Float, nullable=False)
    minutes = Column(Integer, nullable=False)
    weekday = Column(Integer, nullable=False)

    # Relacja do tabeli Profit
    profits = relationship("Profit", back_populates="position")

# Model dla tabeli Profit
class Profit(Base):
    __tablename__ = 'profits'

    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(DateTime, default=dt.now(), nullable=False)
    profit = Column(Float, nullable=False)
    profit_max = Column(Float, nullable=False)
    profit0 = Column(Float, nullable=False)
    mean_profit = Column(Float, nullable=False)
    spread = Column(Float, nullable=False)
    volume_condition = Column(Boolean, nullable=False)
    ticket = Column(Integer, ForeignKey('positions.ticket'), nullable=False)

    # Relacja do tabeli Position
    position = relationship("Position", back_populates="profits")


# Klasa do zarządzania bazą danych
class DatabaseManager:
    def __init__(self, db_url='sqlite:///position_history.db'):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_position(self, ticket, symbol, pos_type, open_time, volume, price_open, comment, reverse_mode, trigger, trigger_divider,
                     decline_factor, profit_factor, calculated_profit, minutes, weekday):
        session = self.Session()

        # Sprawdzenie, czy pozycja z danym ticket już istnieje
        existing_position = session.query(Position).filter_by(ticket=ticket).first()

        if existing_position:
            #print(f"Pozycja z ticketem {ticket} już istnieje. Nie dodano nowego rekordu.")
            pass
        else:
            # Tworzenie nowej pozycji, jeśli nie ma jeszcze pozycji o danym ticket
            new_position = Position(
                ticket=ticket,
                symbol=symbol,
                pos_type=pos_type,
                open_time=open_time,
                volume=volume,
                price_open=price_open,
                comment=comment,
                reverse_mode=reverse_mode,
                trigger=trigger,
                trigger_divider=trigger_divider,
                decline_factor=decline_factor,
                profit_factor=profit_factor,
                calculated_profit=calculated_profit,
                minutes=minutes,
                weekday=weekday
            )
            session.add(new_position)
            session.commit()
            print(f"Dodano nową pozycję z ticketem {ticket}.")

        session.close()

    def add_profit(self, ticket, profit, profit_max, profit0, mean_profit, spread, volume_condition):
        session = self.Session()
        position = session.query(Position).filter_by(ticket=ticket).first()
        if position:
            new_profit = Profit(
                profit=profit,
                profit_max=profit_max,
                profit0=profit0,
                mean_profit=mean_profit,
                spread=spread,
                ticket=ticket,
                volume_condition=volume_condition,
                position=position
            )
            session.add(new_profit)
            session.commit()
        session.close()


class TradingProcessor:
    def __init__(self):
        # Tworzymy instancję klasy DatabaseManager
        self.db_manager = DatabaseManager()

    def process_new_position(self, ticket, symbol, pos_type, open_time, volume, price_open, comment, reverse_mode, trigger,
                             trigger_divider, decline_factor, profit_factor, calculated_profit, minutes, weekday):
        # Przetwarzanie danych pozycji - np. tutaj możesz dodać logikę obliczeń, walidacji itp.
        # Dodajemy pozycję do bazy danych
        self.db_manager.add_position(
            ticket=ticket,
            symbol=symbol,
            pos_type=pos_type,
            open_time=open_time,
            volume=volume,
            price_open=price_open,
            comment=comment,
            reverse_mode=reverse_mode,
            trigger=trigger,
            trigger_divider=trigger_divider,
            decline_factor=decline_factor,
            profit_factor=profit_factor,
            calculated_profit=calculated_profit,
            minutes=minutes,
            weekday=weekday
        )

    def process_profit(self, ticket, profit, profit_max, profit0, mean_profit, spread, volume_condition):
        # Przetwarzanie profitu
        # Dodajemy profit do bazy danych
        self.db_manager.add_profit(ticket=ticket, profit=profit, profit_max=profit_max,
                                   profit0=profit0, mean_profit=mean_profit, spread=spread, volume_condition=volume_condition)


class ReadDatabase:
    def __init__(self, db_url='sqlite:///position_history.db'):
        # Tworzymy silnik
        self.engine = create_engine(db_url)

        # Tworzymy sesję
        self.Session = sessionmaker(bind=self.engine)

    def read_positions_to_df(self):
        """Odczytuje dane z tabeli 'Position' i konwertuje je do pandas DataFrame."""
        query = "SELECT * FROM positions"  # SQL zapytanie do odczytu danych z tabeli 'Position'
        with self.engine.connect() as connection:
            df_positions = pd.read_sql(query, connection)
        return df_positions

    def read_profits_to_df(self):
        """Odczytuje dane z tabeli 'Profit' i konwertuje je do pandas DataFrame."""
        query = "SELECT * FROM profits"  # SQL zapytanie do odczytu danych z tabeli 'Profit'
        with self.engine.connect() as connection:
            df_profits = pd.read_sql(query, connection)
        return df_profits


if __name__=='__main__':
    from result_report import get_raw_close_positions_data
    from numpy import vectorize
    import matplotlib.pyplot as plt
    import numpy as np
    # Przykład użycia
    db_manager = ReadDatabase()

    # Odczyt danych z tabeli 'Position' do pandas DataFrame
    positions_profits = get_raw_close_positions_data(5, -2)
    def give_me_profit(ticket):
        df = positions_profits[positions_profits['position_id']==ticket]
        try:
            return df['profit'].iloc[-1]
        except IndexError:
            return 0

    vectorize_ = vectorize(give_me_profit)

    df_positions = db_manager.read_positions_to_df()
    #df_positions = df_positions[df_positions['weekday'] == 2]
    #df_positions = df_positions[(df_positions['symbol'] != 'XAUUSD') & (df_positions['symbol'] != 'GBPUSD')]
    df_positions['close_profit'] = vectorize_(df_positions['ticket'])
    print(df_positions.close_profit.sum())
    #Odczyt danych z tabeli 'Profit' do pandas DataFrame
    df_profits = db_manager.read_profits_to_df()
    #df_profits = df_profits[df_profits['volume_condition']==False]

    def group_profit_by(what):
        print(f'\n{what}')
        grouped_profit_max = df_profits.groupby('ticket').agg({'profit': 'max'})
        grouped_profit_min = df_profits.groupby('ticket').agg({'profit': 'min'})
        grouped_profit_mean = df_profits.groupby('ticket').agg({'profit': 'mean'})
        grouped_profit_max.reset_index(inplace=True)
        grouped_profit_min.reset_index(inplace=True)
        grouped_profit_mean.reset_index(inplace=True)
        df_positions['max_profit'] = grouped_profit_max['profit'] / df_positions['calculated_profit']
        df_positions['min_profit'] = grouped_profit_min['profit'] / df_positions['calculated_profit']
        df_positions['mean_profit'] = grouped_profit_mean['profit'] / df_positions['calculated_profit']
        a = df_positions.groupby(what).agg(max_profit=('max_profit', 'mean'), std_max=('max_profit', 'std'), std_min=('min_profit', 'std'))
        b = df_positions.groupby(what).agg(min_profit=('min_profit', 'mean'))
        c = df_positions.groupby(what).agg(mean_profit=('mean_profit', 'mean'), counter=('mean_profit', 'count'), close_profit=('close_profit', 'sum'))
        a['min_profit'] = b['min_profit']
        a['mean_profit'] = c['mean_profit']
        a['max-min'] = a['max_profit'] + a['min_profit']
        a['close_profit'] = c['close_profit']
        a['counter'] = c['counter']
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

    group_profit_by(['reverse_mode','trigger'])
    #group_profit_by('symbol')
    plot_profits(df_profits, 'all', 5, condition='')


