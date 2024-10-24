import pandas as pd
import pandas_ta as ta
import numpy as np
import random
import MetaTrader5 as mt
import time
import sys
import os
from datetime import timedelta
from datetime import datetime as dt
from icecream import ic
import xgboost as xgb
from symbols_rank import symbol_stats
from functions import *
from model_generator import data_operations, evening_hour, probability_edge
from parameters import intervals, game_system, reverse_, tz_diff
import random
from database_class import TradingProcessor
processor = TradingProcessor()
mt.initialize()

catalog = os.path.dirname(__file__)
catalog = f'{catalog}\\models'


class Bot:
    weekday = dt.now().weekday()
    tz_diff = tz_diff
    trigger_mode = 'on'
    trigger_model_divider = 9
    profit_factor = 1.5
    position_size = 6       # percent of balance
    kill_multiplier = 1.5   # loss of daily volatility by one position multiplier
    tp_miner = 3
    system = game_system # absolute, weighted_democracy, ranked_democracy, just_democracy
    master_interval = intervals[0]
    decline_factor = 0.65

    def __init__(self, symbol):
        print(dt.now())
        self.change = 0
        self.tiktok = 0
        self.number_of_positions = 0
        self.reverse = reverse_
        self.symbol = symbol
        self.profits = []
        self.profit0 = None
        self.profit_max = 0
        self.fake_position = False
        self.max_close = None
        self.fake_stoploss = 0
        self.fake_counter = 0
        self.fake_stoploss_interval = 'M5'
        self.df_d1 = get_data(symbol, "D1", 1, 30)
        self.avg_daily_vol_()
        self.round_number = round_number_(self.symbol)
        self.volume_calc(Bot.position_size, True)
        self.pos_time = interval_time(Bot.master_interval)
        self.positions_()
        self.trigger = 'model' # 'model' 'moving_averages'
        self.load_models_democracy(catalog)
        self.start_pos = self.pos_type = self.actual_position_democracy()
        self.barOpen = mt.copy_rates_from_pos(self.symbol, timeframe_(self.interval), 0, 1)[0][0]
        printer("Target:", f"{self.tp_miner} $")
        printer("Killer:", f"{-self.kill_position_profit} $")
        self.active_session()

    @class_errors
    def position_time(self):
        dt_from_timestamp = dt.fromtimestamp(mt.positions_get(symbol=self.symbol)[0][1])
        return int((dt.now() - dt_from_timestamp - timedelta(hours=Bot.tz_diff)).seconds/60)

    @class_errors
    def change_trigger_or_reverse(self, what):
        text_ = 'Model of position making is'
        if what == 'trigger':
            self.trigger = 'moving_averages' if self.trigger=='model' else 'model'
            printer(text_, self.trigger)
            self.change = 1
        elif what == 'reverse':
            self.reverse = 'reverse' if self.reverse=='normal' else 'normal'
            printer(text_, self.reverse)
            self.change = 1
        elif what == 'both':
            self.trigger = 'moving_averages' if self.trigger=='model' else 'model'
            printer(text_, self.trigger)
            self.reverse = 'reverse' if self.reverse=='normal' else 'normal'
            printer(text_, self.reverse)
            self.change = 1

    @class_errors
    def if_tiktok(self, profit_=False):
        if self.tiktok <= 3:
            if profit_:
                self.change_trigger_or_reverse('trigger')
                self.tiktok -= 1
            else:
                self.change_trigger_or_reverse('reverse')
                self.tiktok += 1
        else:
            self.change_trigger_or_reverse('both')
            self.tiktok = 0

        self.tiktok = 0 if self.tiktok < 0 else self.tiktok

    @class_errors
    def fake_position_robot(self):
        print("Check fake position")
        interval_df = get_data(self.symbol, self.fake_stoploss_interval, 1, 3)
        close0 = interval_df['close'].iloc[0]
        close1 = interval_df['close'].iloc[1]
        close2 = interval_df['close'].iloc[2]
        pos_type = self.positions[0].type

        def fake_position_on():
            self.fake_position = True
            self.max_close = close2
            self.fake_stoploss = close1

        def fake_position_off():
            self.fake_position = False
            self.max_close = None
            self.fake_stoploss = None
            self.fake_counter = 0
            return self.actual_position_democracy()

        if not self.fake_position:
            if pos_type == 0 and (close2 > close1 > close0):
                fake_position_on()
            elif pos_type == 1 and (close2 < close1 < close0):
                fake_position_on()

        elif self.fake_position and pos_type == 0:
            old_max = self.max_close
            self.max_close = close2 if (close2 > self.max_close) else self.max_close
            if self.max_close > old_max:
                self.fake_counter+=1
                self.fake_stoploss = close1
            if close2 < self.fake_stoploss:
                return fake_position_off()
            else:
                return pos_type

        elif self.fake_position and pos_type == 1:
            old_max = self.max_close
            self.max_close = close2 if (close2 < self.max_close) else self.max_close
            if self.max_close < old_max:
                self.fake_counter+=1
                self.fake_stoploss = close1
            if close2 > self.fake_stoploss:
                return fake_position_off()
            else:
                return pos_type

    @class_errors
    def check_trigger(self, trigger_mode='on'):
        if trigger_mode == 'on':
            position_time = self.position_time()
            try:
                #self.check_volume_condition = True
                profit = sum([i[-4] for i in self.positions])
                if self.profit0 is None:
                    self.profit0 = profit
                self.profits.append(profit+self.profit0)
                self.profit_max = max(self.profits)
                mean_profits = np.mean(self.profits)
                printer("Volume-volatility condition:", self.check_volume_condition)
                printer("Change value:", f"{round(self.profit_needed, 2)} $")
                printer("Actual trigger:", self.trigger)
                printer("Max profit:", f"{self.profit_max} $")
                printer("Profit zero aka spread:", f"{self.profit0} $")
                printer("Mean position profit minus spread:", f"{round(mean_profits-self.profit0, 2)} $")

                if self.fake_position:
                    _ = self.fake_position_robot()

                # Jeżeli strata mniejsza od straty granicznej
                elif profit < -self.profit_needed:
                    self.if_tiktok()

                # Jeżeli strata większa niż strata graniczna podzielona przez współczynnik zysku oraz czas pozycji większy niz czas interwału oraz średni zysk mniejszy niż strata graniczna podzielona przez współczynnik zysku
                elif profit < (-self.profit_needed/Bot.profit_factor) and position_time > self.pos_time and mean_profits < (-self.profit_needed/Bot.profit_factor):
                    self.if_tiktok()

                # Jeżeli zysk większy niż zysk graniczny pomnożony przez współczynnik zysku oraz zysk mniejszy niż zysk maksymalny pomnożony przez współczynik spadku dla danej pozycji i tiktok mniejszy równy 3
                elif self.profit_max > self.profit_needed * Bot.profit_factor and profit < self.profit_max*Bot.decline_factor:
                    self.if_tiktok(True)

                # Jeżeli zysk większy niż zysk graniczny pomnożony przez współczynnik zysku oraz przez 1.5 oraz zysk mniejszy niż zysk maksymalny pomnożony przez powiększony współczynik spadku dla danej pozycji i tiktok mniejszy równy 3
                elif self.profit_max > self.profit_needed * Bot.profit_factor*1.5 and profit < self.profit_max*(((1-Bot.decline_factor)/2)+Bot.decline_factor):
                    self.if_tiktok(True)

                # Jeżeli zysk większy niż zysk graniczny oraz czas pozycji większy niż czas interwału oraz zysk mniejszy niż zysk maksymalny pozycji pomnożony przez współczynnik spadku
                elif self.profit_max > self.profit_needed and position_time > self.pos_time and profit < self.profit_max*Bot.decline_factor:
                    self.if_tiktok(True)

                # Jeżeli zysk większy niż zysk graniczny oraz czas pozycji większy niż czas interwału oraz zysk mniejszy niż zysk maksymalny pozycji pomnożony przez współczynnik spadku
                elif profit > self.profit_needed/ (Bot.profit_factor*1.5) and position_time > self.pos_time/(Bot.profit_factor*1.5):
                    _ = self.fake_position_robot()

                printer("TIKTOK:", self.tiktok)

            except Exception as e:
                print("no positions", e)
                pass
        else:
            pass

    @class_errors
    def positions_(self):
        self.positions = mt.positions_get(symbol=self.symbol)
        if len(self.positions) == 0 and isinstance(self.positions, tuple):
            self.positions = [i for i in self.positions if
                              (i.comment == self.comment)]
            if len(self.positions) == 0:
                self.positions = ()

    @class_errors
    def request_get(self):
        if not self.positions:
            posType = self.actual_position_democracy()
            self.request(actions['deal'], posType)
        self.positions_()

    @class_errors
    def report(self):
        time_sleep = 5
        self.pos_type = self.actual_position_democracy()
        self.positions_()
        # vvv key component vvv
        while True:
            now_ = dt.now()
            if now_.hour >= evening_hour-1:# and now_.minute >= 45:
                self.clean_orders()
                sys.exit()
            self.request_get()
            printer("Symbol:", self.symbol)
            printer("Czas:", time.strftime("%H:%M:%S"))
            self.check_trigger(trigger_mode=Bot.trigger_mode)
            self.data()
            time.sleep(time_sleep)
            print()

    @class_errors
    def data(self, report=True):
        if self.check_new_bar():
            # print(Bot.system)
            self.pos_type = self.actual_position_democracy()
        try:
            act_pos = self.positions[0].type
            if self.pos_type != act_pos:
                self.clean_orders()
        except Exception as e:
            print(e)
            self.clean_orders()

        self.number_of_positions = len(self.positions)
        account = mt.account_info()
        sym_inf = mt.symbol_info(self.symbol)
        act_price = sym_inf.bid
        act_price2 = sym_inf.ask
        profit = sum([i.profit for i in self.positions if
                      ((i.comment == self.comment) and i.magic == self.magic)])
        spread = abs(act_price-act_price2)#real_spread(self.symbol) #*self.number_of_positions*2
        profit_to_margin = round((profit/account.margin)*100, 2)
        profit_to_balance = round((profit/account.balance)*100, 2)

        if report:
            printer(f"Profit:", f"{round(profit, 2)} $")
            printer("Account balance:", f"{account.balance} $")
            printer("Account free margin:", f"{account.margin_free} $")
            printer("Profit to margin:", f"{profit_to_margin} %")
            printer("Profit to balance:", f"{profit_to_balance} %")
            printer("Actual position from model:", f"{self.pos_type}")
            printer("Mode:", f"{self.reverse}")
            printer('Fake position:', self.fake_position)
            printer('Fake counter:', self.fake_counter)
            print()

        # write data to database
        try:
            processor.process_new_position(
                ticket=self.positions[0].ticket,
                symbol=self.symbol,
                pos_type=self.positions[0].type,
                open_time=self.positions[0].time,
                volume=self.positions[0].volume,
                price_open=self.positions[0].price_open,
                comment=self.positions[0].comment,
                reverse_mode=self.reverse,
                trigger=self.trigger,
                trigger_divider=Bot.trigger_model_divider,
                decline_factor=Bot.decline_factor,
                profit_factor=Bot.profit_factor,
                calculated_profit=self.profit_needed,
                minutes=self.pos_time,
                weekday=Bot.weekday
            )

            mean_profit = np.mean(self.profits)
            mean_profit = 0 if mean_profit == np.NaN else mean_profit
            # Dodawanie profitu do istniejącej pozycji
            processor.process_profit(
                ticket=self.positions[0].ticket,
                profit=profit,
                profit_max=self.profit_max,
                profit0=self.profit0,
                mean_profit=mean_profit,
                spread=spread,
                volume_condition=self.check_volume_condition
            )
        except Exception as e:
            print(e)
            pass

        if profit < -self.kill_position_profit:
            print('Loss is to high. I have to kill it!')
            self.clean_orders()
        elif profit > self.tp_miner:
            print('The profit is nice. I want it on our accout.')
            self.clean_orders()

    @class_errors
    def reset_bot(self):
        self.pos_type = None
        self.positions = None
        self.profits = []
        self.profit0 = None
        self.profit_max = 0

    @class_errors
    def clean_orders(self):
        self.close_request()
        orders = mt.orders_get(symbol=self.symbol)
        print(orders)
        counter = 0
        if orders == ():
            print("Brak zleceń oczekujących dla symbolu", self.symbol)
            self.reset_bot()
            self.report()
        else:
            for order in orders:
                request = {
                    "action": mt.TRADE_ACTION_REMOVE,
                    "order": order.ticket,
                    "symbol": order.symbol,
                }
                result = mt.order_send(request)
                if result.retcode != mt.TRADE_RETCODE_DONE:
                    print("Błąd podczas usuwania zlecenia:", result.comment)
                else:
                    print(f"Usunięto zlecenie oczekujące: {order}\n\n")
                    counter += 1

            print(f"Usunięto łącznie {counter} zleceń na symbolu {self.symbol}")
            # time_sleep = int(random.randint(5, 15)*60)
            # print(f"Break {int(time_sleep/60)} minutes.")
            time.sleep(5)
            self.reset_bot()
            self.report()

    @class_errors
    def avg_daily_vol_(self):
        df = self.df_d1
        df['avg_daily'] = (df.high - df.low) / df.open
        self.avg_vol = df['avg_daily'].mean()

    @class_errors
    def volume_calc(self, max_pos_margin: int, min_volume: int) -> None:
        leverage = mt.account_info().leverage
        symbol_info = mt.symbol_info(self.symbol)._asdict()
        price = mt.symbol_info_tick(self.symbol)._asdict()
        margin_min = round(((symbol_info["volume_min"] *
                        symbol_info["trade_contract_size"])/leverage) *
                        price["bid"], 2)
        account = mt.account_info()._asdict()
        max_pos_margin = round(account["balance"] * (max_pos_margin/100) /
                            (self.avg_vol * 100))
        if "JP" not in self.symbol:
            volume = round((max_pos_margin / margin_min)) *\
                            symbol_info["volume_min"]
            printer('Volume from value:', round((max_pos_margin / margin_min), 2))
        else:
            volume = round((max_pos_margin * 100 / margin_min)) *\
                            symbol_info["volume_min"]
            printer('Volume from value:', round((max_pos_margin * 100 / margin_min), 2))
        if volume > symbol_info["volume_max"]:
            volume = float(symbol_info["volume_max"])
        printer('Min volume:', min_volume)
        printer('Calculated volume:', volume)
        self.volume = volume
        if min_volume and (volume < symbol_info["volume_min"]):
            self.volume = symbol_info["volume_min"]
        _, self.kill_position_profit, _ = symbol_stats(self.symbol, self.volume, Bot.kill_multiplier)
        self.tp_miner = round(self.kill_position_profit * Bot.tp_miner / Bot.kill_multiplier, 2)
        self.profit_needed = round(self.kill_position_profit/Bot.trigger_model_divider, 2)

    @class_errors
    def find_files(self, directory):
        """
        Znajduje pliki w danym folderze, których nazwy zawierają określone słowo kluczowe.

        Args:
        - directory (str): Ścieżka do folderu, w którym mają być przeszukiwane pliki.
        - symbol (str): Słowo kluczowe, które ma występować w nazwach plików.

        Returns:
        - list: Lista plików, których nazwy zawierają określone słowo kluczowe.
        """
        matching_files = []
        for filename in os.listdir(directory):
            if self.symbol in filename:
                matching_files.append(filename[:-6].split('_')[:-1])
        df = pd.DataFrame(matching_files, columns=['strategy', 'ma_fast', 'ma_slow',
            'learning_rate', 'training_set', 'symbol', 'interval', 'factor', 'result'])
        df['ma_fast'] = df['ma_fast'].astype(int)
        df['ma_slow'] = df['ma_slow'].astype(int)
        df['factor'] = df['factor'].astype(int)
        df['result'] = df['result'].astype(int)
        df = df.sort_values(by='result', ascending=False)[::2]
        df.reset_index(drop=True, inplace=True)
        df['rank'] = df.index + 1
        _ = df['rank'].to_list()
        _.reverse()
        df['rank'] = _
        # print(df)
        printer("Ilość modeli:", len(df))
        if len(df) < 5:
            print(f"Za mało modeli --> ({len(df)})")
            input("Wciśnij cokolwek żeby wyjść.")
            sys.exit(1)
        names = []
        create_df = []
        for i in range(0, len(df)):
            strategy = df['strategy'].iloc[i]
            ma_fast = df['ma_fast'].iloc[i]
            ma_slow = df['ma_slow'].iloc[i]
            learning_rate = df['learning_rate'].iloc[i]
            training_set = df['training_set'].iloc[i]
            interval = df['interval'].iloc[i]
            factor = df['factor'].iloc[i]
            result = df['result'].iloc[i]
            if Bot.system != 'invertedrank_democracy':
                rank = df['rank'].iloc[i]
            else:
                rank = df['rank'].iloc[len(df)-i-1]
            name = f'{strategy}_{ma_fast}_{ma_slow}_{learning_rate}_{training_set}_{self.symbol}_{interval}_{factor}_{result}'
            names.append([name, rank]) # change tuple to list
            create_df.append(f'{name}'.split('_'))

        if Bot.system == 'weighted_democracy':
            df_result_filter = pd.DataFrame(create_df, columns=['strategy', 'ma_fast', 'ma_slow',
                    'learning_rate', 'training_set', 'symbol', 'interval', 'factor', 'result']
                    )
            df_result_filter['result'] = df_result_filter['result'].astype(int)
            range_ = len(create_df)
            for n in range(range_):
                sum_ = int(df_result_filter.copy().drop(range(n, len(df_result_filter)))['result'].sum()/2)
                result_ = df_result_filter['result'].iloc[n]
                if result_ > sum_:
                    old_list = names[n][0].split('_')
                    index_ = int(len(df_result_filter)/2)-3 if int(len(df_result_filter)/2) > 6 else int(len(df_result_filter)/2)-1
                    old_list[-1] = str(df_result_filter['result'].iloc[index_])
                    new_str = '_'.join(old_list)
                    self.rename_files_in_directory(names[n][0], new_str)
                    names[n][0] = new_str
                else:
                    break
        return names

    @class_errors
    def load_models_democracy(self, directory):
        model_names = self.find_files(directory)
        # buy
        self.buy_models = []
        self.sell_models = []
        self.factors = []
        ma_list = []
        for model_name in model_names:
            model_path_buy = os.path.join(directory, f'{model_name[0]}_buy.model')
            model_path_sell = os.path.join(directory, f'{model_name[0]}_sell.model')
            model_buy = xgb.Booster()
            model_sell = xgb.Booster()
            model_buy.load_model(model_path_buy)
            model_sell.load_model(model_path_sell)
            self.buy_models.append((model_buy, f'{model_name[0]}_{model_name[1]}'))
            self.sell_models.append((model_sell, f'{model_name[0]}_{model_name[1]}'))
            self.factors.append(int(model_name[0].split('_')[-2])) # factor
            ma_list.append((int(model_name[0].split('_')[-8]), int(model_name[0].split('_')[-7])))
        assert len(self.buy_models) == len(self.sell_models)
        # class return
        most_common_ma = most_common_value(ma_list)
        self.ma_factor_fast = most_common_ma[0]
        self.ma_factor_slow = most_common_ma[1]
        self.interval = Bot.master_interval
        self.comment = 'wdemo_4'
        self.magic = magic_(self.symbol, self.comment)
        self.mdv = self.MDV_() / 4

        printer("MA values:", f"fast={self.ma_factor_fast}, slow={self.ma_factor_slow}")
        printer('comment:', self.comment)

    @class_errors
    def actual_position_democracy(self):
        if self.fake_position:
            return self.fake_position_robot()

        if self.trigger == 'model':
            stance_values = []
            i = 0
            for mbuy, msell, factor in zip(self.buy_models, self.sell_models, self.factors):
                df = get_data_for_model(self.symbol, mbuy[1].split('_')[-4], 1, int(factor**2 + 250)) # how_many_bars
                df = data_operations(df, factor)
                dfx = df.copy()
                dtest_buy = xgb.DMatrix(df)
                dtest_sell = xgb.DMatrix(df)
                buy = mbuy[0].predict(dtest_buy)
                sell = msell[0].predict(dtest_sell)
                buy = np.where(buy > probability_edge, 1, 0)
                sell = np.where(sell > probability_edge, -1, 0)
                dfx['stance'] = buy + sell
                dfx['stance'] = dfx['stance'].replace(0, np.NaN)
                dfx['stance'] = dfx['stance'].ffill()

                # if Bot.system == 'just_democracy':
                #     position_ = dfx['stance'].iloc[-1]
                # elif Bot.system =='weighted_democracy':
                position_ = dfx['stance'].iloc[-1] * int(mbuy[1].split('_')[-2])
                # elif Bot.system == 'ranked_democracy':
                #     position_ = dfx['stance'].iloc[-1] * int(mbuy[1].split('_')[-1])
                # elif Bot.system == 'invertedrank_democracy':
                #     position_ = dfx['stance'].iloc[-1] * int(mbuy[1].split('_')[-1])
                try:
                    _ = int(position_)
                except Exception:
                    continue
                stance_values.append(int(position_))
                i+=1
                if i >= 60:
                    break

            print('Stances: ', stance_values)
            sum_of_position = np.sum(stance_values)

            def longs(stance_values):
                return round(np.sum([1 for i in stance_values if i > 0]) / len(stance_values), 2)

            def longs_democratic(stance_values):
                all_ = [abs(i) for i in stance_values]
                return round(np.sum(stance_values) / np.sum(all_), 2)

            force_of_democratic = ic(longs_democratic(stance_values))
            printer("Force of long democratic votes:", force_of_democratic)
            try:
                fx = ic(longs(stance_values))
                printer("Percent of long votes:", fx)
            except Exception:
                pass
            if sum_of_position != 0:
                position = 0 if sum_of_position > 0 else 1
            else:
                try:
                    position = self.pos_type
                except Exception as e:
                    print(e)
                    self.pos_type = self.pos_creator()

            volume_10 = ((df['high']-df['low'])*df['volume']).rolling(8).mean().iloc[-1]
            volume_2 = ((df['high']-df['low'])*df['volume']).rolling(2).mean().iloc[-1]
            print(f"Vol 10: {round(volume_10, 2)} Vol 2: {round(volume_2, 2)}")
            self.check_volume_condition = volume_2 > volume_10

        else:
            printer('Position from moving averages:', f'fast={self.ma_factor_fast} slow={self.ma_factor_slow}')
            dfx = get_data_for_model(self.symbol, self.interval, 1, int(self.ma_factor_slow + 100)) # how_many_bars
            dfx['adj'] = (dfx['close'] + dfx['high'] + dfx['low']) / 3
            ma1 = dfx.ta.sma(length=self.ma_factor_fast)
            ma2 = ta.sma(dfx['adj'], length=self.ma_factor_slow)
            dfx['stance'] = np.where(ma1>=ma2, 1, 0)
            dfx['stance'] = np.where(ma1<ma2, -1, dfx['stance'])
            position = 0 if dfx.stance.iloc[-1] == 1 else 1
            printer("MA position:", position)
            volume_10 = ((dfx['high']-dfx['low'])*dfx['volume']).rolling(8).mean().iloc[-1]
            volume_2 = ((dfx['high']-dfx['low'])*dfx['volume']).rolling(2).mean().iloc[-1]
            print(f"Vol 10: {round(volume_10, 2)} Vol 2: {round(volume_2, 2)}")
            self.check_volume_condition = volume_2 > volume_10

        if self.reverse == 'normal':
            pass
        elif self.reverse == 'reverse':
            position = 0 if position == 1 else 1
        elif self.reverse == 'normal_mix':
            time_ = dt.now()
            if time_.hour >= 14:
                position = 0 if position == 1 else 1

        return position

    @class_errors
    def rename_files_in_directory(self, old_phrase, new_phrase):
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

    @class_errors
    def request(self, action, posType, price=None):
        self.volume_calc(Bot.position_size, True)
        if action == actions['deal']:
            print("YES")
            price = mt.symbol_info(self.symbol).bid
        else:
            if posType == 0:
                posType = pendings["long_stop"]
            else:
                posType = pendings["short_stop"]

        request = {
            "action": action,
            "symbol": self.symbol,
            "volume": self.volume,
            "type": posType,
            "price": float(price),
            "deviation": 20,
            "magic": self.magic,
            "tp": 0.0,
            "sl": 0.0,
            "comment": self.comment,
            "type_time": mt.ORDER_TIME_GTC,
            "type_filling": mt.ORDER_FILLING_IOC,
            }
        order_result = mt.order_send(request)
        print(order_result)

    @class_errors
    def delete_model(self):
        os.remove(self.model_buy[1])
        print(f"Model removed: {self.model_buy[1]}")
        os.remove(self.model_sell[1])
        print(f"Model removed: {self.model_sell[1]}")

    @class_errors
    def check_new_bar(self):
        if self.change == 0:
            bar = mt.copy_rates_from_pos(
                self.symbol, timeframe_(self.interval), 0, 1) # change self.interval to 'M1'
            if self.barOpen == bar[0][0]:
                return False
            else:
                self.barOpen = bar[0][0]
                return True
        else:
            self.change = 0
            return True

    @class_errors
    def MDV_(self):
        """Returns mean daily volatility"""
        df = self.df_d1.copy()
        df['mean_vol'] = (df.high - df.low)
        return df['mean_vol'].mean()

    @class_errors
    def active_session(self):
        #from model_generator import morning_hour
        df = get_data(self.symbol, 'D1', 0, 1)
        today_date_tuple = time.localtime()
        formatted_date = time.strftime("%Y-%m-%d", today_date_tuple)
        if str(df.time[0].date()) == formatted_date:
            pass
        else:
            print(f"Session on {self.symbol} is not active")
            input()
            sys.exit(1)

    @class_errors
    def close_request(self):
        positions_ = mt.positions_get(symbol=self.symbol)
        for i in positions_:
            request = {"action": mt.TRADE_ACTION_DEAL,
                        "symbol": i.symbol,
                        "volume": float(i.volume),
                        "type": 1 if (i.type == 0) else 0,
                        "position": i.ticket,
                        "magic": i.magic,
                        'deviation': 20,
                        "type_time": mt.ORDER_TIME_GTC,
                        "type_filling": mt.ORDER_FILLING_IOC
                        }
            order_result = mt.order_send(request)

    @class_errors
    def pos_creator(self):
        try:
            return self.pos_type
        except NameError:
            return random.randint(0, 1)


if __name__ == '__main__':
    print('Yo, wtf?')