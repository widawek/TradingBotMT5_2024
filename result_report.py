import MetaTrader5 as mt
import pandas as pd
from datetime import datetime as dt
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
mt.initialize()


def pandas_options():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_colwidth', None)

pandas_options()

def get_raw_close_postions_data(from_when: int, to_when: int = -1):
    from_date = dt.today().date() - timedelta(days=from_when)
    print(from_date)
    to_date = dt.today().date() - timedelta(days=to_when)
    print(f"Data from {from_date.strftime('%A')} {from_date} to {to_date.strftime('%A')} {to_date}")
    from_date = dt(from_date.year, from_date.month, from_date.day)
    to_date = dt(to_date.year, to_date.month, to_date.day)
    data = mt.history_deals_get(from_date, to_date)
    df = pd.DataFrame(list(data), columns=data[0]._asdict().keys())
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[df['symbol'] != '']
    df['profit'] = df['profit'] + df['commission'] * 2 + df['swap']
    df['reason'] = df['reason'].shift(1)
    df = df.drop(columns=['time_msc', 'commission', 'external_id', 'swap'])
    df = df[df['time'].dt.date >= from_date.date()]
    df = df.sort_values(by=['position_id', 'time'])
    df.reset_index(drop=True, inplace=True)
    df['profit'] = df['profit'].shift(-1)
    df['time_close'] = df['time'].shift(-1)
    df = df.rename(columns={'time': 'time_open'})
    df['sl'] = df.comment.shift(-1).str.contains('sl', na=False)
    df['tp'] = df.comment.shift(-1).str.contains('tp', na=False)
    df = df.iloc[::2]
    df['hour_open'] = df['time_open'].dt.hour
    df['hour_close'] = df['time_close'].dt.hour
    df['weekday'] = df['time_close'].dt.day_name()
    df['plus'] = df['profit'] > 0
    df['minus'] = df['profit'] < 0
    df['plus'] = df['plus'].astype(int)
    df['minus'] = df['minus'].astype(int)
    df.reset_index(drop=True, inplace=True)
    #print(df)
    return df


def plot_results(from_when: int,
                 to_when: int=-1,
                 by_: str='symbol',
                 profitable_only: bool=False,
                 percent_of_balance_for_po: float=0,
                 plot: bool=True,
                 non_profitable_only: bool=False):
    """
    Generates a bar plot for profits grouped by a given category within a
    specified time range.

    Parameters:
    from_when (int): The starting time for the data range.
    to_when (int): The ending time for the data range. Default is -1 (indicating the latest data).
    by_ (str): The column name to group by and plot. Default is 'symbol'.
    profitable_only (bool): If True, only include rows where profit exceeds a certain percentage of the account balance. Default is False.
    percent_of_balance_for_po ([int, float]): The percentage of the account balance used as a threshold for filtering profitable rows. Default is 0.

    Returns:
    None
    """
    try:
        df = get_raw_close_postions_data(from_when, to_when)
    except IndexError:
        return []
    #df = df[df['symbol'] != 'EURGBP']
    margin = mt.account_info().balance
    print(f"Actual balance: {margin}")
    print("RR: ", round(df['plus'].sum()/df['minus'].sum(), 2))
    df = df.groupby(by_)['profit'].sum().reset_index()
    if profitable_only:
        df = df[df['profit'] > margin*percent_of_balance_for_po/100]
    elif non_profitable_only:
        df = df[df['profit'] < margin*percent_of_balance_for_po/100]

    if by_ == 'weekday':
        sort_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        df['weekday'] = pd.Categorical(df['weekday'], categories=sort_order, ordered=True)
        df = df.sort_values('weekday')
    print(df)
    if plot:
        plt.figure(figsize=(10, 6))
        plt.bar(df[by_], df['profit'], color='skyblue')
        # Dodanie tytułu i etykiet
        plt.title('Raport Zysków dla Symboli', fontsize=16)
        plt.xlabel(by_, fontsize=14)
        plt.ylabel('Zysk', fontsize=14)
        # Dodanie siatki
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        # Wyświetlenie wykresu
        plt.tight_layout()
        plt.show()

    if by_ == 'symbol':
        x1 = sorted(df.symbol.to_list())
        print(x1, f'\nNumber of symbols: {len(x1)}')
        print(df['profit'].sum())
        return x1
    return []

if __name__ == "__main__":
    # by_ 'symbol', 'comment', 'interval', 'factor', 'learing_rate', 'training_set'
    x1 = plot_results(0, -2)
    print(x1)