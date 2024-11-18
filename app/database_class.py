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
Base = declarative_base()

# Pobieramy bieżący katalog roboczy
current_working_dir = os.getcwd()
db_path = os.path.join(current_working_dir, 'database', 'position_history.db')
db_url = f"sqlite:///{db_path}"

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
    trend = Column(String, nullable=False)
    tiktok = Column(Integer, nullable=False)
    number_of_models = Column(Integer, nullable=False)
    market = Column(String, nullable=False)
    full_reverse = Column(Boolean, nullable=False)

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
    fake_position = Column(Boolean, nullable=False)
    fake_position_counter = Column(Integer, nullable=False)
    fake_position_stoploss = Column(Float, nullable=False)

    # Relacja do tabeli Position
    position = relationship("Position", back_populates="profits")


# Klasa do zarządzania bazą danych
class DatabaseManager:
    def __init__(self, db_url = f"sqlite:///{db_path}"):
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def add_position(self, ticket, symbol, pos_type, open_time, volume, price_open, comment, reverse_mode, trigger, trigger_divider,
                     decline_factor, profit_factor, calculated_profit, minutes, weekday, trend, tiktok, number_of_models, market,
                     full_reverse):
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
                weekday=weekday,
                trend=trend,
                tiktok=tiktok,
                number_of_models=number_of_models,
                market=market,
                full_reverse=full_reverse
            )
            session.add(new_position)
            session.commit()
            print(f"Dodano nową pozycję z ticketem {ticket}.")

        session.close()

    def add_profit(self, ticket, profit, profit_max, profit0, mean_profit, spread, volume_condition,
                   fake_position,fake_position_counter,fake_position_stoploss):
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
                position=position,
                fake_position=fake_position,
                fake_position_counter=fake_position_counter,
                fake_position_stoploss=fake_position_stoploss
            )
            session.add(new_profit)
            session.commit()
        session.close()


class TradingProcessor:
    def __init__(self):
        # Tworzymy instancję klasy DatabaseManager
        self.db_manager = DatabaseManager()

    def process_new_position(self, ticket, symbol, pos_type, open_time, volume, price_open, comment, reverse_mode, trigger,
                             trigger_divider, decline_factor, profit_factor, calculated_profit, minutes, weekday, trend, tiktok,
                             number_of_models, market, full_reverse):
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
            weekday=weekday,
            trend=trend,
            tiktok=tiktok,
            number_of_models=number_of_models,
            market=market,
            full_reverse=full_reverse
        )

    def process_profit(self, ticket, profit, profit_max, profit0, mean_profit, spread, volume_condition,
                       fake_position,fake_position_counter,fake_position_stoploss):
        # Przetwarzanie profitu
        # Dodajemy profit do bazy danych
        self.db_manager.add_profit(
            ticket=ticket,
            profit=profit,
            profit_max=profit_max,
            profit0=profit0,
            mean_profit=mean_profit,
            spread=spread,
            volume_condition=volume_condition,
            fake_position=fake_position,
            fake_position_counter=fake_position_counter,
            fake_position_stoploss=fake_position_stoploss)


class ReadDatabase:
    def __init__(self, db_url= f"sqlite:///{db_path}"):
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
    print("YO WTF?")


