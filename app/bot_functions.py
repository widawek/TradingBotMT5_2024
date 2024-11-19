import os
from datetime import datetime as dt
from datetime import timedelta
import MetaTrader5 as mt
import pandas as pd
mt.initialize()


def rename_files_in_directory(old_phrase, new_phrase, catalog):
    # Iterate over all files in the specified directory
    for filename in os.listdir(catalog):
        # Check if the old phrase is in the filename
        if old_phrase in filename:
            # Create the new filename by replacing the old phrase with the new one
            new_filename = filename.replace(old_phrase, new_phrase)
            # Construct the full old and new file paths
            old_file_path = os.path.join(catalog, filename)
            new_file_path = os.path.join(catalog, new_filename)

            def change_(old_file_path, new_file_path):
                os.rename(old_file_path, new_file_path)
                #print(f'Renamed: {old_file_path} -> {new_file_path}')
            # Rename the file
            try:
                change_(old_file_path, new_file_path)
            except FileExistsError:
                new_file_path = new_file_path.split('_')
                result = int(new_file_path[-2])
                new_file_path[-2] = str(result + 1)
                new_file_path = '_'.join(new_file_path)
                change_(old_file_path, new_file_path)


def checkout_report(symbol, reverse, trigger, condition):
    from_date = dt.today().date() - timedelta(days=0)
    print(from_date)
    to_date = dt.today().date() + timedelta(days=1)
    print(f"Data from {from_date.strftime('%A')} {from_date} to {to_date.strftime('%A')} {to_date}")
    from_date = dt(from_date.year, from_date.month, from_date.day)
    to_date = dt(to_date.year, to_date.month, to_date.day)
    try:
        data = mt.history_deals_get(from_date, to_date)
        df = pd.DataFrame(list(data), columns=data[0]._asdict().keys())
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df[(df['symbol'] != '') & (df['symbol']==symbol)]
        df['profit'] = df['profit'] + df['commission'] * 2 + df['swap']
        df['reason'] = df['reason'].shift(1)
        df = df.drop(columns=['time_msc', 'commission', 'external_id', 'swap'])
        df = df[df['time'].dt.date >= from_date.date()]
        df = df[df.groupby('position_id')['position_id'].transform('count') == 2]
        df = df.sort_values(by=['position_id', 'time'])
        df.reset_index(drop=True, inplace=True)
        df['profit'] = df['profit'].shift(-1)
        df['time_close'] = df['time'].shift(-1)
        df = df.rename(columns={'time': 'time_open'})
        df['sl'] = df.comment.shift(-1).str.contains('sl', na=False)
        df['tp'] = df.comment.shift(-1).str.contains('tp', na=False)
        df = df.iloc[::2]

        if len(df) < 3:
            return False

        prof_lst = df['profit'][-3:].to_list()
        comm_lst = df['comment'][-3:].to_list()
        if comm_lst[0][0] == reverse[0]:
            if comm_lst[0][2] == trigger[-1]:

                # BotReverse
                if condition:
                    return all([all([i>0 for i in prof_lst]),
                            all([i==comm_lst[0] for i in comm_lst])])

                return all([all([i<0 for i in prof_lst]),
                            all([i==comm_lst[0] for i in comm_lst])])
    except Exception as e:
        print(e)
        return False

    return False
