import pandas as pd
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
from extensions.symbols_rank import symbol_stats
from app.functions import *
from app.decorators import class_errors
from app.model_generator import data_operations, evening_hour, probability_edge
from config.parameters import *
from app.database_class import TradingProcessor
from app.bot_functions import *
sys.path.append("..")



processor = TradingProcessor()
mt.initialize()

catalog = os.path.dirname(__file__)
parent_catalog = os.path.dirname(catalog)
catalog = f'{parent_catalog}\\models'


class Bot:
    weekday = dt.now().weekday()

    def __init__(self, symbol):
        self.changer_reverse = False
        self.interval = master_interval if symbol not in ['AUDUSD'] else 'M10'
        printer(dt.now(), symbol)
        self.symbol = symbol
        self.magic = magic_(symbol, 'bot_2024')
        self.model_counter = None
        self.profit0 = None
        self.max_close = None
        self.fresh_signal = None
        self.strategy_pos_open_price = None
        self.good_price_to_open_pos = None
        self.base_fake_interval = base_fake_interval
        self.print_count = 0
        self.change = 0
        self.tiktok = 0
        self.number_of_positions = 0
        self.profit_max = 0
        self.fake_stoploss = 0
        self.fake_counter = 0
        self.fake_position = False
        self.profits = []
        self.close_profits = []
        self.global_positions_stats = []
        self.trigger = 'model' # 'model' 'moving_averages'
        self.market = 'e' if dt.now().hour < change_hour else 'u'
        self.trend = 'neutral' # long_strong, long_weak, long_normal, short_strong, short_weak, short_normal, neutral
        self.trigger_model_divider = avg_daily_vol_for_divider(symbol, trigger_model_divider_factor)
        self.trend_or_not = trend_or_not(symbol)
        self.reverse = reverse_(symbol)
        self.df_d1 = get_data(symbol, "D1", 1, 30)
        self.avg_daily_vol()
        self.round_number = round_number_(symbol)
        self.volume_calc(position_size, True)
        self.pos_time = interval_time(self.interval)
        self.positions_()
        self.load_models_democracy(catalog)
        self.barOpen = mt.copy_rates_from_pos(symbol, timeframe_(self.interval), 0, 1)[0][0]
        self.active_session()

    @class_errors
    def position_time(self):
        try:
            dt_from_timestamp = dt.fromtimestamp(mt.positions_get(symbol=self.symbol)[0][1])
        except Exception:
            return 0
        return int((dt.now() - dt_from_timestamp - timedelta(hours=tz_diff)).seconds/60)

    @class_errors
    def change_trigger_or_reverse(self, what):
        text_ = 'Model of position making is'
        match what:
            case 'trigger':
                self.trigger = changer(self.trigger, 'moving_averages', 'model')
            case 'reverse':
                self.reverse = changer(self.reverse, 'reverse', 'normal')
            case 'both':
                self.trigger = changer(self.trigger, 'moving_averages', 'model')
                self.reverse = changer(self.reverse, 'reverse', 'normal')
        printer(text_, f'{self.trigger}, {self.reverse}')
        self.change = 1

    @class_errors
    def if_tiktok(self):
        pos = mt.positions_get(symbol=self.symbol)
        profit_ = sum([pos[i].profit for i in range(len(pos)) if pos[i].magic == self.magic])
        self.close_profits.append((profit_, self.comment[:-1]))
        if len(self.close_profits) >= 2:
            try:
                last_to_by_comment = [i[0] for i in profit_ if i[1] == self.comment[:-1]]
                if len(last_to_by_comment) >= 2:
                    last_two = sum(last_to_by_comment[-2:])
                    printer("Last two positions profit", f"{last_two:.2f} $")
                else:
                    last_two = 0
            except Exception:
                last_two = 0
        else:
            last_two = 0

        if self.tiktok < 3:
            if profit_ > 0 and last_two >= 0:
                self.tiktok -= 1
            elif (profit_ < 0 and self.tiktok in [0, 2]) or (last_two < 0):
                self.tiktok += 1
            elif (profit_ < 0 and self.tiktok == 1) or (last_two < 0):
                self.change_trigger_or_reverse('trigger')
                self.tiktok += 1
            else:
                pass
        else:
            if profit_ > 0 and last_two >= 0:
                self.tiktok -= 1
            else:
                self.change_trigger_or_reverse('both')
                self.tiktok = 0

        self.tiktok = 0 if self.tiktok < 0 else self.tiktok

    @class_errors
    def fake_position_robot(self):
        if self.fake_counter <= 5:
            interval = self.base_fake_interval
        elif self.fake_counter <= 8:
            interval = 'M5'
        elif self.fake_counter <= 11:
            interval = 'M10'
        elif self.fake_counter <= 14:
            interval = 'M15'
        elif self.fake_counter <= 16:
            interval = 'M30'

        interval_df = get_data(self.symbol, interval, 1, 3)
        close0 = interval_df['close'].iloc[0]
        close1 = interval_df['close'].iloc[1]
        close2 = interval_df['close'].iloc[2]
        pos_type = self.positions[0].type
        profit_ = self.positions[0].profit
        try:
            if self.base_fake_interval != interval:
                test = get_data(self.symbol, interval, 1, 200)
                test['better_close'] = np.where(((test['close']>test['close'].shift(1)) & (test['close']>test['close'].shift(2)) & (pos_type==0)) |
                                                ((test['close']<test['close'].shift(1)) & (test['close']<test['close'].shift(2)) & (pos_type==1)), 1, 0)
                test_better = test[test['better_close']==1]
                test_better.reset_index()
                self.fake_stoploss = test_better['close'].iloc[-2]
                self.base_fake_interval = interval
                return pos_type
        except Exception as e:
            print(e)
            pass


        def fake_position_on():
            self.fake_position = True
            self.max_close = close2
            self.fake_stoploss = close1

        def fake_position_off():
            self.fake_position = False
            self.max_close = None
            self.fake_stoploss = 0
            self.fake_counter = 0
            self.base_fake_interval = base_fake_interval
            return self.actual_position_democracy()

        if not self.fake_position:
            if (((pos_type == 0) and (close2 > close1 > close0)) or\
                ((pos_type == 1) and (close2 < close1 < close0))) and\
                    (profit_ > 0):
                fake_position_on()

        elif self.fake_position and pos_type == 0:
            old_max = self.max_close
            self.max_close = close2 if (close2 > self.max_close) else self.max_close
            if self.max_close > old_max:
                self.fake_counter+=1
                self.fake_stoploss = close1
            if (close2 < self.fake_stoploss) or (profit_ < 0):
                return fake_position_off()
            else:
                return pos_type

        elif self.fake_position and pos_type == 1:
            old_max = self.max_close
            self.max_close = close2 if (close2 < self.max_close) else self.max_close
            if self.max_close < old_max:
                self.fake_counter+=1
                self.fake_stoploss = close1
            if (close2 > self.fake_stoploss) or (profit_ < 0):
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

                # BotReverse
                if reverse_it_all:
                    profit = -profit

                if self.profit0 is None:
                    self.profit0 = profit
                self.profits.append(profit+self.profit0)
                self.profit_max = max(self.profits)
                self.profit_min = min(self.profits)
                mean_profits = np.mean(self.profits)
                self.self_decline_factor()
                if self.print_condition():
                    printer("Volume-volatility condition:", self.check_volume_condition)
                    printer("Change value:", f"{round(self.profit_needed, 2):.2f} $")
                    printer("Actual trigger:", self.trigger)
                    printer("Max profit:", f"{self.profit_max:.2f} $")
                    printer("Profit zero aka spread:", f"{self.profit0:.2f} $")
                    printer("Mean position profit minus spread:", f"{round(mean_profits-self.profit0, 2):.2f} $")
                    printer("Decline factor:", f"{self.profit_decline_factor}")

                if self.fake_position:
                    _ = self.fake_position_robot()

                # Jeżeli strata mniejsza od straty granicznej
                elif profit < -self.profit_needed and profit > 0.91 * self.profit_min:
                    self.clean_orders()

                # Jeżeli strata większa niż strata graniczna podzielona przez współczynnik zysku oraz czas pozycji większy niz czas interwału oraz średni zysk mniejszy niż strata graniczna podzielona przez współczynnik zysku
                elif (profit < (-self.profit_needed/profit_factor)) \
                    and (position_time > self.pos_time) \
                    and (mean_profits < (-self.profit_needed/profit_factor)):
                    self.clean_orders()

                # Jeżeli zysk większy niż zysk graniczny pomnożony przez współczynnik zysku oraz zysk mniejszy niż zysk maksymalny pomnożony przez współczynik spadku dla danej pozycji i tiktok mniejszy równy 3
                elif (self.profit_max > self.profit_needed) \
                    and (profit < self.profit_max*self.profit_decline_factor):
                    self.clean_orders()
                # Jeżeli zysk większy niż zysk graniczny oraz czas pozycji większy niż czas interwału oraz zysk mniejszy niż zysk maksymalny pozycji pomnożony przez współczynnik spadku
                elif (profit > self.profit_needed/(profit_factor*1.5) and not reverse_it_all) or \
                    (profit < -self.profit_needed/(profit_factor*1.5) and reverse_it_all):
                    #and (position_time > self.pos_time/(profit_factor*1.5))
                    _ = self.fake_position_robot()

                if self.print_condition():
                    printer("TIKTOK:", self.tiktok)

            except Exception as e:
                print("no positions", e)
                pass
        else:
            pass

    @class_errors
    def clean_orders(self):
        self.if_tiktok()
        self.close_request()
        orders = mt.orders_get(symbol=self.symbol)
        print(orders)
        counter = 0
        if orders == ():
            print("Brak zleceń oczekujących dla symbolu", self.symbol)
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
            time.sleep(1)
        self.reset_bot()
        self.report()

    @class_errors
    def print_condition(self):
        return ((self.print_count == 0) or (self.print_count % 10 == 0))

    @class_errors
    def self_decline_factor(self, multiplier: int=3):
        min_val = 0.45
        max_val = 0.85
        min_value = 0
        max_value = self.profit_needed*multiplier
        # only variable is self.profit_max
        normalized_value = (self.profit_max - min_value) / (max_value - min_value)
        exp_value = np.exp(normalized_value) - 1
        exp_max = np.exp(1) - 1
        #return
        self.profit_decline_factor = round(min_val + (exp_value / exp_max) * (max_val - min_val), 3)
        if self.profit_decline_factor > max_val:
            self.profit_decline_factor = max_val

    @class_errors
    def positions_(self):
        self.positions = mt.positions_get(symbol=self.symbol)
        if len(self.positions) == 0 and isinstance(self.positions, tuple):
            self.positions = [i for i in self.positions if
                              (i.magic == self.magic)]
            if len(self.positions) == 0:
                self.positions = ()

    @class_errors
    def request_get(self):
        if not self.positions:
            checkout_report(self.symbol, self.reverse, self.trigger, reverse_it_all)
            posType = self.actual_position_democracy()
            self.request(actions['deal'], posType)
        self.positions_()

    @class_errors
    def report(self):
        time_sleep = 2
        self.pos_type = self.actual_position_democracy()
        self.positions_()
        # vvv key component vvv
        while True:
            if dt.now().hour >= evening_hour:# and now_.minute >= 45:
                self.clean_orders()
                sys.exit()
            self.request_get()
            if self.print_condition():
                printer("\nSymbol:", self.symbol)
                printer("Czas:", time.strftime("%H:%M:%S"))
            self.check_trigger(trigger_mode=trigger_mode)
            self.data()
            time.sleep(time_sleep)

    @class_errors
    def data(self, report=True):
        profit = sum([i.profit for i in self.positions if
            ((i.comment == self.comment) and (i.magic == self.magic))])
        if self.check_new_bar():
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
        spread = abs(act_price-act_price2)
        try:
            profit_to_margin = round((profit/account.margin)*100, 2)
        except ZeroDivisionError:
            print("WTF??")
            profit_to_margin = 0
        profit_to_balance = round((profit/account.balance)*100, 2)
        if report:
            if self.print_condition():
                self.info(profit, account, profit_to_margin, profit_to_balance)
        self.print_count += 1

        self.write_to_database(profit, spread)

        if profit < -self.kill_position_profit:
            print('Loss is to high. I have to kill it!')
            self.clean_orders()
        elif profit > self.tp_miner:
            print('The profit is nice. I want it on our accout.')
            self.clean_orders()

    @class_errors
    def info(self, profit, account, profit_to_margin, profit_to_balance):
        printer("Profit:", f"{round(profit, 2):.2f} $")
        printer("Account balance:", f"{account.balance:.2f} $")
        printer("Account free margin:", f"{account.margin_free:.2f} $")
        printer("Profit to margin:", f"{profit_to_margin:.2f} %")
        printer("Profit to balance:", f"{profit_to_balance:.2f} %")
        printer("Actual position from model:", self.pos_type)
        printer("Mode:", self.reverse)
        printer("Fake position:", self.fake_position)
        printer("Fake counter:", self.fake_counter)
        printer("Trend:", self.trend)
        print()

    @class_errors
    def reset_bot(self):
        self.pos_type = None
        self.positions = None
        self.profits = []
        self.profit0 = None
        self.profit_max = 0

    @class_errors
    def avg_daily_vol(self):
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
        self.volume = volume
        if min_volume and (volume < symbol_info["volume_min"]):
            self.volume = symbol_info["volume_min"]
        _, self.kill_position_profit, _ = symbol_stats(self.symbol, self.volume, kill_multiplier)
        self.kill_position_profit = round(self.kill_position_profit, 2)# * (1+self.multi_voltage('M5', 33)), 2)
        self.tp_miner = round(self.kill_position_profit * tp_miner / kill_multiplier, 2)
        self.profit_needed = round(self.kill_position_profit/self.trigger_model_divider, 2)
        self.reverse_full_reverse_from_reverse_to_no_reverse_but_no_reverse_position_only_check()
        printer('Min volume:', min_volume)
        printer('Calculated volume:', volume)
        printer("Target:", f"{self.tp_miner:.2f} $")
        printer("Killer:", f"{-self.kill_position_profit:.2f} $")


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
        df = pd.DataFrame(matching_files, columns=['market', 'strategy', 'ma_fast', 'ma_slow',
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
        self.model_counter = len(df[df['market'] == self.market])
        printer("Ilość modeli:", self.model_counter)
        if self.model_counter > max_number_of_models:
            self.model_counter = max_number_of_models
        if len(df) < 5:
            print(f"Za mało modeli --> ({self.model_counter})")
            input("Wciśnij cokolwek żeby wyjść.")
            sys.exit(1)
        names = []
        create_df = []
        for i in range(0, len(df)):
            market = df['market'].iloc[i]
            strategy = df['strategy'].iloc[i]
            ma_fast = df['ma_fast'].iloc[i]
            ma_slow = df['ma_slow'].iloc[i]
            learning_rate = df['learning_rate'].iloc[i]
            training_set = df['training_set'].iloc[i]
            interval = df['interval'].iloc[i]
            factor = df['factor'].iloc[i]
            result = df['result'].iloc[i]
            if game_system != 'invertedrank_democracy':
                rank = df['rank'].iloc[i]
            else:
                rank = df['rank'].iloc[len(df)-i-1]
            name = f'{market}_{strategy}_{ma_fast}_{ma_slow}_{learning_rate}_{training_set}_{self.symbol}_{interval}_{factor}_{result}'
            names.append([name, rank])
            create_df.append(f'{name}'.split('_'))

        if game_system == 'weighted_democracy':
            df_result_filter = pd.DataFrame(create_df, columns=['market', 'strategy', 'ma_fast', 'ma_slow',
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
                    rename_files_in_directory(names[n][0], new_str, catalog)
                    names[n][0] = new_str
                else:
                    break
        return names

    @class_errors
    def load_models_democracy(self, directory):
        self.trigger = 'model'
        self.reverse = reverse_(self.symbol)
        self.tiktok = 0
        self.change = 0
        model_names = self.find_files(directory)
        self.buy_models = []
        self.sell_models = []
        self.factors = []
        ma_list = []
        for model_name in model_names:
            if model_name[0].split('_')[-10] == self.market:
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
            else:
                continue
        assert len(self.buy_models) == len(self.sell_models)

        most_common_ma = most_common_value(ma_list)
        self.ma_factor_fast = most_common_ma[0]
        self.ma_factor_slow = most_common_ma[1]
        self.mdv = self.mdv_() / 4
        printer("MA values:", f"fast={self.ma_factor_fast}, slow={self.ma_factor_slow}")

    @class_errors
    def actual_position_democracy(self, number_of_bars=250):
        try:
            if self.fake_position:
                return self.fake_position_robot()

            market = 'e' if dt.now().hour < change_hour else 'u'
            if market != self.market:
                self.market = market
                self.load_models_democracy(catalog)

            if self.trigger == 'model':
                stance_values = []
                dataframes = []
                i = 0
                start = time.time()
                for mbuy, msell, factor in zip(self.buy_models, self.sell_models, self.factors):
                    name_ = f"{mbuy[1].split('_')[-4]}_{factor}"
                    if i==0:
                        df = get_data_for_model(self.symbol, mbuy[1].split('_')[-4], 1, int(factor**2 + number_of_bars)) # how_many_bars
                        df = data_operations(df, factor)
                        dataframes.append((df, name_))
                    else:
                        if any(nazwa == name_ for _, nazwa in dataframes):
                            for dataframe, nazwa in dataframes:
                                if nazwa == name_:
                                    df = dataframe
                                    break
                        else:
                            df = get_data_for_model(self.symbol, mbuy[1].split('_')[-4], 1, int(factor**2 + number_of_bars)) # how_many_bars
                            df = data_operations(df, factor)
                            dataframes.append((df, name_))
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
                    position_ = dfx['stance'].iloc[-1] * int(mbuy[1].split('_')[-2])
                    try:
                        _ = int(position_)
                    except Exception:
                        continue
                    stance_values.append(int(position_))
                    i+=1
                    if i >= self.model_counter:
                        break
                end = time.time()
                duration = end-start
                time_info(duration, "Decision time")
                del dataframes
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
                    position = self.pos_creator()

                volume_10 = ((df['high']-df['low'])*df['volume']).rolling(8).mean().iloc[-1]
                volume_2 = ((df['high']-df['low'])*df['volume']).rolling(2).mean().iloc[-1]
                print(f"Vol 10: {round(volume_10, 2)} Vol 2: {round(volume_2, 2)}")
                self.check_volume_condition = volume_2 > volume_10

            else:
                dfx = get_data_for_model(self.symbol, self.interval, 1, int(self.ma_factor_slow + 100)) # how_many_bars
                dfx["stance"] = function_when_model_not_work(dfx, self.ma_factor_fast, self.ma_factor_slow)
                position = 0 if dfx.stance.iloc[-1] == 1 else 1
                volume_10 = ((dfx['high']-dfx['low'])*dfx['volume']).rolling(8).mean().iloc[-1]
                volume_2 = ((dfx['high']-dfx['low'])*dfx['volume']).rolling(2).mean().iloc[-1]
                printer('Position from moving averages:', f'fast={self.ma_factor_fast} slow={self.ma_factor_slow}')
                printer("MA position:", position)
                print(f"Vol 10: {round(volume_10, 2)} Vol 2: {round(volume_2, 2)}")
                self.check_volume_condition = volume_2 > volume_10

            if self.reverse == 'normal':
                pass
            elif self.reverse == 'reverse':
                position = changer(position, 0, 1)
            elif self.reverse == 'normal_mix':
                time_ = dt.now()
                if time_.hour >= 14:
                    position = changer(position, 0, 1)

            if not self.changer_reverse:
                # BotReverse
                if reverse_it_all:
                    position = changer(position, 0, 1)

            # finding the last opening price as strategy_pos_open_price
            dfx['cross'] = np.where(dfx['stance'] != dfx['stance'].shift(), 1, 0)
            self.fresh_signal = True if dfx['stance'].iloc[-1] != dfx['stance'].iloc[-2] else False
            cross = dfx[dfx['cross'] == 1]
            self.strategy_pos_open_price = cross['open'].iloc[-1]

            positions_ = mt.positions_get(symbol=self.symbol)
            if len(positions_) == 0:
                while True:
                    if self.fresh_signal:
                        break

                    # check if price is nice to open
                    tick = mt.symbol_info_tick(self.symbol)
                    price = round((tick.ask + tick.bid) / 2, self.round_number)
                    match position:
                        case 0: self.good_price_to_open_pos = True if price <= self.strategy_pos_open_price else False
                        case 1: self.good_price_to_open_pos = True if price >= self.strategy_pos_open_price else False
                    if self.good_price_to_open_pos:
                        break

                    if self.check_new_bar():
                        return self.actual_position_democracy(number_of_bars=number_of_bars)
                    pos = 'LONG' if position==0 else "SHORT"
                    diff = round((price - self.strategy_pos_open_price) * 100 / self.strategy_pos_open_price, 2)
                    printer('Symbol / Position / difference', f'{self.symbol} / {pos} / {diff:.2f} %', base_just=65)
                    time.sleep(5)

        except KeyError:
            return self.actual_position_democracy(number_of_bars=number_of_bars*2)

        return position

    @class_errors
    def what_trend_is_it(self, posType):
        smallest = int(position_size/2)
        smaller = int(position_size/1.5)
        normal = position_size
        bigger = int(position_size*1.5)
        biggest = int(position_size*2)
        wow = int(position_size*3)

        self.trend = vwap_std(self.symbol)

        # BotReverse
        if reverse_it_all:
            posType = changer(posType, 0, 1)

        if posType == (0 if not self.trend_or_not else 1):
            match self.trend:
                case 'sold_out': return wow  # price is low
                case 'long_strong': return normal  # price is high
                case 'long_normal': return smaller
                case 'long_weak': return bigger  # price is low
                case 'long_super_weak': return biggest  # price is low
                case 'short_strong': return normal  # price is low
                case 'short_normal': return smaller
                case 'short_weak': return smaller  # price is high
                case 'short_super_weak': return smallest

        elif posType == (1 if not self.trend_or_not else 0):
            match self.trend:
                case 'overbought': return wow
                case 'long_strong': return normal
                case 'long_normal': return smaller
                case 'long_weak': return smaller
                case 'long_super_weak': return smallest
                case 'short_strong': return normal
                case 'short_normal': return smaller
                case 'short_weak': return bigger
                case 'short_super_weak': return biggest

        return position_size

    @class_errors
    def request(self, action, posType, price=None):
        if action == actions['deal']:
            print("YES")
            price = mt.symbol_info(self.symbol).bid
        else:
            if posType == 0:
                posType = pendings["long_stop"]
            else:
                posType = pendings["short_stop"]

        position_size = self.what_trend_is_it(posType)
        self.volume_calc(position_size, True)
        if self.trend_or_not:
            letter = "t"
        else:
            letter = "f"
        self.comment = f'{self.reverse[0]}_{self.trigger[-1]}_{self.ma_factor_fast}_{self.ma_factor_slow}_{letter}_{self.market}_{self.tiktok}'

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
    def mdv_(self):
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
            request = {
                "action": mt.TRADE_ACTION_DEAL,
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

    @class_errors
    def write_to_database(self, profit, spread):
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
                trigger_divider=self.trigger_model_divider,
                decline_factor=self.profit_decline_factor,
                profit_factor=profit_factor,
                calculated_profit=self.profit_needed,
                minutes=self.pos_time,
                weekday=Bot.weekday,
                trend=self.trend,
                tiktok=self.tiktok,
                number_of_models = self.model_counter,
                market=self.market,
                full_reverse=reverse_it_all
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
                volume_condition=self.check_volume_condition,
                fake_position=self.fake_position,
                fake_position_counter=self.fake_counter,
                fake_position_stoploss=self.fake_stoploss
            )
        except Exception as e:
            print(e)
            pass

    # def multi_voltage(self, interval, factor):
    #     df = get_data(self.symbol, interval, 1, factor*3)
    #     df['atr'] = df.ta.atr(length=factor)
    #     df['atr_ma'] = df.ta.atr(length=factor*2)
    #     df['atr_vol_1'] = df['atr'] * df['volume'].rolling(window=factor).mean()
    #     df['atr_vol_2'] = df['atr_ma'] * df['volume'].rolling(window=factor*2).mean()
    #     df['xxx'] = (1 - (df['atr_vol_1'] / df['atr_vol_2']))#*-1
    #     return df['xxx'].iloc[-1]


    @class_errors
    def reverse_full_reverse_from_reverse_to_no_reverse_but_no_reverse_position_only_check(self):
        if dt.now().hour >= morning_hour + 3:
            if not self.changer_reverse:
                try:
                    from_date = dt.today().date()
                    to_date = dt.today().date() + timedelta(days=1)
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
                    df = df[df.groupby('position_id')['position_id'].transform('count') == 2]
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
                    df = df[df['symbol'] == self.symbol]
                    df.reset_index(drop=True, inplace=True)
                    df = df.groupby('hour_close').agg(
                        profit=('profit', 'sum'),
                        count_sum=('profit', 'size')
                    ).reset_index()
                    df = df.sort_values('hour_close')
                    result = df['profit'].rolling(3).mean()
                    result = float(result.iloc[-1])
                    print(result)
                except Exception as e:
                    print(e)
                try:
                    if isinstance(result, float):
                        if result < 0:
                            self.changer_reverse = True
                            print("CHANGER WAS CHANGED")
                except Exception as e:
                    print(e)
            else:
                print("NOT CHECK")



if __name__ == '__main__':
    print('Yo, wtf?')
