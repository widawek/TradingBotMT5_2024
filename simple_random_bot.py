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
import math
from icecream import ic
import xgboost as xgb
from symbols_rank import symbol_stats
from functions import *
from model_generator import data_operations, evening_hour, probability_edge
from parameters import intervals, game_system, reverse_, tz_diff
from dataclasses import dataclass
import random

catalog = os.path.dirname(__file__)
catalog = f'{catalog}\\models'


class Bot:
    tz_diff = tz_diff
    trigger_mode = 'on'
    sl_mdv_multiplier = 1.5 # mdv multiplier for sl
    tp_mdv_multiplier = 2   # mdv multiplier for tp
    position_size = 6       # percent of balance
    kill_multiplier = 1.5   # loss of daily volatility by one position multiplier
    tp_miner = 3
    time_limit_multiplier = 3
    system = game_system # absolute, weighted_democracy, ranked_democracy, just_democracy
    master_interval = intervals[0]
    factor_to_delete = 24

    def __init__(self, symbol, _, symmetrical_positions, daily_volatility_reduce):
        self.change = 0
        self.tiktok = 0
        print(dt.now())
        mt.initialize()
        self.reverse = reverse_
        self.symbol = symbol
        self.df_d1 = get_data(symbol, "D1", 1, 30)
        self.avg_daily_vol_()
        self.round_number = round_number_(self.symbol)
        self.symmetrical_positions = symmetrical_positions
        self.daily_volatility_reduce_values = daily_volatility_reduce[0]
        self.max_reduce = daily_volatility_reduce[1]
        self.min_reduce = daily_volatility_reduce[2]
        self.volume_calc(Bot.position_size, True)
        self.number_of_positions = 0
        self.positions_()
        self.limits = None
        self.sl_positions = None
        self.sl = 0.0
        self.tp = 0.0
        self.trigger = 'model' # 'model' 'moving_averages'
        #self.reverse_mechanism = CheckOpenPositions(symbol, Bot.master_interval, self.kill_position_profit)
        if not 'democracy' in Bot.system:
            self.load_models(catalog)  # initialize few class variables
            self.start_pos = self.pos_type = self.actual_position()
            self.barOpen = mt.copy_rates_from_pos(self.symbol, timeframe_(self.interval), 0, 1)[0][0]
        else:
            self.load_models_democracy(catalog)  # initialize few class variables
            self.start_pos = self.pos_type = self.actual_position_democracy()
            self.barOpen = mt.copy_rates_from_pos(self.symbol, timeframe_(self.interval), 0, 1)[0][0]
        print("Target == ", self.tp_miner, " USD")
        print("Killer == ", -self.kill_position_profit, " USD")
        self.active_session()

    @class_errors
    def position_time(self):
        dt_from_timestamp = dt.fromtimestamp(mt.positions_get(symbol=self.symbol)[0][1])
        return int((dt.now() - dt_from_timestamp - timedelta(hours=Bot.tz_diff)).seconds/60)

    @class_errors
    def change_trigger_or_reverse(self, what):
        if what == 'trigger':
            self.trigger = 'moving_averages' if self.trigger=='model' else 'model'
            print(f'Model of position making is {self.trigger}')
            self.change = 1
        elif what == 'reverse':
            self.reverse = 'reverse' if self.reverse=='normal' else 'normal'
            print(f'Model positioning system is {self.reverse}')
            self.change = 1
        elif what == 'both':
            self.trigger = 'moving_averages' if self.trigger=='model' else 'model'
            print(f'Model of position making is {self.trigger}')
            self.reverse = 'reverse' if self.reverse=='normal' else 'normal'
            print(f'Model positioning system is {self.reverse}')
            self.change = 1

    @class_errors
    def check_trigger(self, trigger_mode='on'):
        if trigger_mode == 'on':
            print("Trigger check!!!")
            trigger_model_divider = 20
            profit_ = round(self.tp_miner/trigger_model_divider, 2)
            profit_factor = 1.5
            minutes = interval_time(Bot.master_interval)
            position_time = self.position_time()
            try:
                #self.check_volume_condition = True
                print("Volume-volatility condition: ", self.check_volume_condition)
                profit = sum([i[-4] for i in self.positions])
                print(f"Check trigger profit = {round(profit, 2)} USD")
                print(f"Change value = {round(profit_, 2)} USD")
                print(f"Actual trigger = {self.trigger}")

                if profit > profit_ and self.tiktok <= 3:
                    self.change_trigger_or_reverse('trigger')
                    self.tiktok += 1
                elif profit > profit_ and self.tiktok > 3:
                    self.change_trigger_or_reverse('both')
                    self.tiktok = 0
                elif profit < -profit_ * profit_factor and self.tiktok <= 3 and position_time > minutes:
                    self.change_trigger_or_reverse('trigger')
                    self.tiktok += 1
                elif profit < -profit_ * profit_factor and self.tiktok > 3 and position_time > minutes:
                    self.change_trigger_or_reverse('both')
                    self.tiktok = 0
                print("TIKTOK: ", self.tiktok)

            except Exception as e:
                print("no positions", e)
                pass
        else:
            pass

    @class_errors
    def positions_test(self):
        positions = mt.positions_get(symbol=self.symbol)
        positions = [i for i in positions if
                        (i.comment == self.comment)]
        if self.positions is not None:
            if len(positions) < len(self.positions) and len(positions) > 0:
                self.clean_orders()

    @class_errors
    def positions_(self):
        self.positions = mt.positions_get(symbol=self.symbol)
        if len(self.positions) == 0 and isinstance(self.positions, tuple):
            self.positions = [i for i in self.positions if
                              (i.comment == self.comment)]
            if len(self.positions) == 0:
                self.positions = ()

    @class_errors
    def scan_orders(self, orders, price):
        rel_tol = 1e-10 * 10**(8 - self.round_number)
        math_true_false_list = [math.isclose(i.price_open, price,
                                                rel_tol=rel_tol, abs_tol=rel_tol)
                                                for i in orders]
        return any(math_true_false_list)

    @class_errors
    def request_get(self):
        if self.positions:
            posType = self.positions[0].type
            price_open = self.positions[0].price_open
            if self.limits is None:
                _, self.limits = self.prices_list(posType, price_open=price_open)
                print(self.limits)
            orders = mt.orders_get(symbol=self.symbol)
            self.positions_test()

            if posType == 0:
                price_ = mt.symbol_info(self.symbol).bid
                for idx, i in enumerate(self.limits):
                    p_ = round(i+self.mdv, self.round_number)
                    if price_ < i:
                        if self.scan_orders(orders, p_):
                            pass
                        else:
                            self.request(actions['pending'], posType, price=p_)
                            self.limits.remove(self.limits[idx])
                            break
            else:
                price_ = mt.symbol_info(self.symbol).bid
                for idx, i in enumerate(self.limits):
                    p_ = round(i-self.mdv, self.round_number)
                    if price_ > i:
                        if self.scan_orders(orders, p_):
                            pass
                        else:
                            self.request(actions['pending'], posType, price=p_)
                            self.limits.remove(self.limits[idx])
                            break

            self.sl_giver()
            self.tp_giver()
            self.change_tp_sl()

        else:
            if Bot.system == 'absolute':
                self.check_model_()
                posType = self.actual_position()
            else:
                posType = self.actual_position_democracy()
            stops, _ = self.prices_list(posType)
            print(stops)
            self.request(actions['deal'], posType)
            for i in stops:
                self.request(actions['pending'], posType, i)

        self.positions_()

    @class_errors
    def report(self):
        if Bot.system == 'absolute':
            time_sleep = 5
            self.pos_type = self.actual_position()
        else:
            time_sleep = 10
            self.pos_type = self.actual_position_democracy()
        self.positions_()
        # vvv key component vvv
        while True:
            now_ = dt.now()
            if now_.hour >= evening_hour-1:# and now_.minute >= 45:
                self.clean_orders()
                sys.exit()
            self.request_get()
            print("Czas:", time.strftime("%H:%M:%S"))
            self.check_trigger(trigger_mode=Bot.trigger_mode)
            self.data()
            time.sleep(time_sleep)
            print()

    @class_errors
    def data(self, report=True):
        if self.check_new_bar():
            if 'democracy' not in Bot.system:
                print("One model")
                self.pos_type = self.actual_position()
            else:
                print(Bot.system)
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
        act_price = mt.symbol_info(self.symbol).bid
        profit = sum([i.profit for i in self.positions if
                      ((i.comment == self.comment) and i.magic == self.magic)])
        mean_open_price = np.mean([i.price_open for i in self.positions if
                      ((i.comment == self.comment) and i.magic == self.magic)])
        spread = real_spread(self.symbol) #*self.number_of_positions*2
        profit_to_margin = round((profit/account.margin)*100, 2)

        if self.pos_type == 0:
            distance = round(((act_price - mean_open_price) / mean_open_price)*100, 2)
            self.zero_point = round(mean_open_price + spread, self.round_number)
            self.barrier_price = round(self.zero_point + Bot.sl_mdv_multiplier * self.mdv, self.round_number)
        elif self.pos_type == 1:
            distance = round(((mean_open_price - act_price) / mean_open_price)*100, 2)
            self.zero_point = round(mean_open_price - spread, self.round_number)
            self.barrier_price = round(self.zero_point - Bot.sl_mdv_multiplier * self.mdv, self.round_number)
        else:
            distance = 'Unknown'
            self.zero_point = 'Unknown'
            self.barrier_price = 'Unknown'

        if report:
            print(f"MDV:                                              {round(self.mdv, self.round_number)}")
            print(f"RMR Strategy for symbol {str(self.symbol).ljust(6)} profit:            {round(profit, 2)} $")
            print(f"number of positions:                              {self.number_of_positions}")
            print(f"Account balance:                                  {account.balance} $")
            print(f"Account free margin:                              {account.margin_free} $")
            print(f"Actual zero profit price:                         {self.zero_point}")
            print(f"Profit to margin:                                 {profit_to_margin} %")
            print(f"Percent distance from actual price to mean price: {distance} %")
            print(f"Distance to daily volatility:                     {round(distance/self.avg_vol, 2)} %")
            print(f"Actual price:                                     {act_price}")
            print(f"Barrier price:                                    {self.barrier_price}")
            print(f"Spread:                                           {spread}")
            print(f"Actual position from model:                       {self.pos_type}")
            print(f"Mode:                                             {self.reverse}")
            print()

        if profit < -self.kill_position_profit:
            print('Loss is to high. I have to kill it!')
            self.clean_orders()
        elif profit > self.tp_miner:
            print('The profit is nice. I want it on our accout.')
            self.clean_orders()

    @class_errors
    def reset_bot(self):
        self.tp = 0.0
        self.sl = 0.0
        self.limits = None
        self.pos_type = None
        self.positions = None
        self.time_stp = dt.now()

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
            print('Volume form: ', (max_pos_margin / margin_min))
        else:
            volume = round((max_pos_margin * 100 / margin_min)) *\
                            symbol_info["volume_min"]
            print('Volume form: ', (max_pos_margin * 100 / margin_min))
        if volume > symbol_info["volume_max"]:
            volume = float(symbol_info["volume_max"])
        print('Min volume: ', min_volume)
        print('Calculated volume: ', volume)
        self.volume = volume
        if min_volume and (volume < symbol_info["volume_min"]):
            self.volume = symbol_info["volume_min"]
        _, self.kill_position_profit, _ = symbol_stats(self.symbol, self.volume, Bot.kill_multiplier)
        self.tp_miner = round(self.kill_position_profit * Bot.tp_miner / Bot.kill_multiplier, 2)

    @class_errors
    def check_model_(self):
        time_now = dt.now()
        act_prof = mt.account_info().profit
        act_prof = 0 if act_prof < 0 else act_prof
        if time_now - timedelta(minutes=self.limit_time) > self.time_stp:
            from_date = dt.today() - timedelta(days=1)
            to_date = dt.today() + timedelta(days=1)
            from_date = dt(from_date.year, from_date.month, from_date.day)
            to_date = dt(to_date.year, to_date.month, to_date.day)
            data = mt.history_deals_get(from_date, to_date)
            try:
                df = pd.DataFrame(list(data), columns=data[0]._asdict().keys())
            except IndexError:
                pass
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df[df['profit'] != 0.0]
            df_limit_time = df["time"].iloc[-1]
            limit_time = df_limit_time - timedelta(minutes=self.limit_time)
            df = df[df['time'] > limit_time]
            if len(df) > 0:
                profit = df['profit'].sum() + act_prof
            else:
                profit = 0

            if profit >= 0:
                self.time_stp = dt.now()
                self.limit_time = int(self.limit_time*1.33)
            else:
                self.delete_model()
                self.load_models(catalog)
                self.clean_orders()

    @class_errors
    def check_model_democracy(self):
        #shift = self.reverse_mechanism.time_shift
        time_now = dt.now()
        act_prof = mt.account_info().profit
        act_prof = 0 if act_prof < 0 else act_prof
        if time_now - timedelta(minutes=self.limit_time) > self.time_stp:
            from_date = dt.today() - timedelta(days=1)
            to_date = dt.today() + timedelta(days=1)
            from_date = dt(from_date.year, from_date.month, from_date.day)
            to_date = dt(to_date.year, to_date.month, to_date.day)
            data = mt.history_deals_get(from_date, to_date)
            try:
                df = pd.DataFrame(list(data), columns=data[0]._asdict().keys())
            except IndexError:
                pass
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df[df['profit'] != 0.0]
            df_limit_time = df["time"].iloc[-1]
            limit_time = df_limit_time - timedelta(minutes=self.limit_time)
            df = df[df['time'] > limit_time]
            if len(df) > 0:
                profit = df['profit'].sum() + act_prof
            else:
                profit = 0

            if profit >= 0:
                self.time_stp = dt.now()
                self.limit_time = int(self.limit_time*1.33)
            else:
                self.delete_model()
                self.load_models(catalog)
                self.clean_orders()

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
        print(df)
        if len(df) < 5:
            print(f"Za mało modeli --> ({len(df)})")
            input("Wciśnij cokolwek żeby wyjść.")
            sys.exit(1)
        if Bot.system == 'absolute':
            strategy = df['strategy'].iloc[0]
            ma_fast = df['ma_fast'].iloc[0]
            ma_slow = df['ma_slow'].iloc[0]
            learning_rate = df['learning_rate'].iloc[0]
            training_set = df['training_set'].iloc[0]
            interval = df['interval'].iloc[0]
            factor = df['factor'].iloc[0]
            result = df['result'].iloc[0]
            name = f'{strategy}_{ma_fast}_{ma_slow}_{learning_rate}_{training_set}_{self.symbol}_{interval}_{factor}_{result}'
            return name
        else:
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
            print(names)

            if Bot.system == 'weighted_democracy':
                print(names)
                df_result_filter = pd.DataFrame(create_df, columns=['strategy', 'ma_fast', 'ma_slow',
                        'learning_rate', 'training_set', 'symbol', 'interval', 'factor', 'result']
                        )
                print(df_result_filter)
                df_result_filter['result'] = df_result_filter['result'].astype(int)
                range_ = len(create_df)
                for n in range(range_):
                    sum_ = int(df_result_filter.copy().drop(range(n, len(df_result_filter)))['result'].sum()/2)
                    result_ = df_result_filter['result'].iloc[n]
                    print("sum", sum_)
                    print("result", result_)
                    if result_ > sum_:
                        old_list = names[n][0].split('_')
                        #print("OLD: ", old_list)
                        index_ = int(len(df_result_filter)/2)-3 if int(len(df_result_filter)/2) > 6 else int(len(df_result_filter)/2)-1
                        old_list[-1] = str(df_result_filter['result'].iloc[index_])
                        new_str = '_'.join(old_list)
                        self.rename_files_in_directory(names[n][0], new_str)
                        names[n][0] = new_str
                        # print("change names")#
                        # print(names)
                    else:
                        print('break')
                        break

            print(names)
            return names

    @class_errors
    def load_models(self, directory):
        model_name = self.find_files(directory)
        print(model_name)
        model_path_buy = os.path.join(directory, f'{model_name}_buy.model')
        model_path_sell = os.path.join(directory, f'{model_name}_sell.model')
        model_buy = xgb.Booster()
        model_sell = xgb.Booster()
        model_buy.load_model(model_path_buy)
        model_sell.load_model(model_path_sell)
        mod_buy = [model_buy, model_path_buy]
        mod_sell = [model_sell, model_path_sell]
        self.time_stp = dt.now()
        _string_data = mod_buy[1].split('_')
        self.factor = _string_data[-3]
        self.ts = _string_data[-6]
        if len(self.ts) < 2:
            self.ts = self.ts + '0'
        self.lr = _string_data[-7][-2:]
        if len(self.lr) < 2:
            self.lr = self.lr + '0'
        self.limit_time = interval_time(self.interval) * Bot.time_limit_multiplier
        self.model_buy = mod_buy
        self.model_sell = mod_sell
        self.daily_volatility_reducer()
        self.comment = f'{self.lr}_{self.ts}_{self.interval}_{self.factor}_{self.daily_volatility_reduce}'
        self.magic = magic_(self.symbol, self.comment)
        self.mdv = self.MDV_() / self.daily_volatility_reduce

    @class_errors
    def actual_position(self):
        # Przykładowe użycie:
        df = get_data_for_model(self.symbol, self.interval, 1, 800)
        df = data_operations(df, 10)
        dfx = df.copy()
        dtest_buy = xgb.DMatrix(df)
        dtest_sell = xgb.DMatrix(df)
        buy = self.model_buy[0].predict(dtest_buy)
        sell = self.model_sell[0].predict(dtest_sell)
        buy = np.where(buy > probability_edge, 1, 0)
        sell = np.where(sell > probability_edge, -1, 0)
        dfx['stance'] = buy + sell
        dfx['stance'] = dfx['stance'].replace(0, np.NaN)
        dfx['stance'] = dfx['stance'].ffill()
        position = 0 if dfx['stance'].iloc[-1] == 1 else 1
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
    def load_models_democracy(self, directory):
        model_names = self.find_files(directory)
        #print(model_names)
        # buy
        self.buy_models = []
        self.sell_models = []
        self.factors = []
        ma_list = []
        for model_name in model_names:
            print(model_name)
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
        self.limit_time = interval_time(self.interval) * Bot.time_limit_multiplier
        self.daily_volatility_reducer()
        y_ = Bot.system.split('_')[1][:4]
        self.comment = f'{Bot.system[0]+y_}_{self.daily_volatility_reduce}'
        self.magic = magic_(self.symbol, self.comment)
        self.mdv = self.MDV_() / self.daily_volatility_reduce

        print(f"MA values: fast={self.ma_factor_fast}, slow={self.ma_factor_slow}")
        print('comment: ', self.comment)
        print('Democracy')

    @class_errors
    def actual_position_democracy(self):

        if self.trigger == 'model':
            stance_values = []
            for mbuy, msell, factor in zip(self.buy_models, self.sell_models, self.factors):
                df = get_data_for_model(self.symbol, mbuy[1].split('_')[-4], 1, int(factor**2 + 500)) # how_many_bars
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

                if Bot.system == 'just_democracy':
                    position_ = dfx['stance'].iloc[-1]
                elif Bot.system =='weighted_democracy':
                    position_ = dfx['stance'].iloc[-1] * int(mbuy[1].split('_')[-2])
                elif Bot.system == 'ranked_democracy':
                    position_ = dfx['stance'].iloc[-1] * int(mbuy[1].split('_')[-1])
                elif Bot.system == 'invertedrank_democracy':
                    position_ = dfx['stance'].iloc[-1] * int(mbuy[1].split('_')[-1])
                try:
                    _ = int(position_)
                except Exception:
                    continue
                stance_values.append(int(position_))

            print('Stances: ', stance_values)
            sum_of_position = np.sum(stance_values)

            def longs(stance_values):
                return round(np.sum([1 for i in stance_values if i > 0]) / len(stance_values), 2)

            def longs_democratic(stance_values):
                all_ = [abs(i) for i in stance_values]
                return round(np.sum(stance_values) / np.sum(all_), 2)

            force_of_democratic = ic(longs_democratic(stance_values))
            print("Force of long democratic votes: ", force_of_democratic)
            try:
                fx = ic(longs(stance_values))
                print("Percent of long votes: ", fx)
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
            print(f'Position from moving averages fast={self.ma_factor_fast} slow={self.ma_factor_slow}')
            dfx = get_data_for_model(self.symbol, self.interval, 1, int(self.ma_factor_slow + 100)) # how_many_bars
            dfx['adj'] = (dfx['close'] + dfx['high'] + dfx['low']) / 3
            ma1 = dfx.ta.sma(length=self.ma_factor_fast)
            ma2 = ta.sma(dfx['adj'], length=self.ma_factor_slow)
            dfx['stance'] = np.where(ma1>=ma2, 1, 0)
            dfx['stance'] = np.where(ma1<ma2, -1, dfx['stance'])
            position = 0 if dfx.stance.iloc[-1] == 1 else 1
            print("MA position: ", position)
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
                    print(f'Renamed: {old_file_path} -> {new_file_path}')
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
    def tp_giver(self):
        if self.number_of_positions < self.symmetrical_positions:
            pass
        else:
            act_price = mt.symbol_info(self.symbol).bid
            last_position_open_time = pd.to_datetime(self.positions[-1].time, unit='s')
            df = get_data(self.symbol, 'M1', 0, 1440)
            df = df[df['time'] > last_position_open_time]
            if self.pos_type == 0:
                highest_price = df.high.max()
                if act_price < (highest_price - self.mdv * Bot.sl_mdv_multiplier / 3):
                    new_tp = round(highest_price + self.mdv * Bot.tp_mdv_multiplier,
                                   self.round_number)
                    self.tp = new_tp if new_tp > self.barrier_price else self.tp
            elif self.pos_type == 1:
                lowest_price = df.low.min()
                if act_price > (lowest_price + self.mdv * Bot.sl_mdv_multiplier / 3):
                    new_tp = round(lowest_price - self.mdv * Bot.tp_mdv_multiplier,
                                   self.round_number)
                    self.tp = new_tp if new_tp < self.barrier_price else self.tp
            print("Actual TP == {}".format(self.tp))

    @class_errors
    def clean_sl(self):
        act_price = mt.symbol_info(self.symbol).bid
        if self.sl == 0.0:
            if self.pos_type == 0:
                if act_price > self.barrier_price:
                    self.sl = round(
                        self.zero_point + (0.1*self.mdv), self.round_number)
                    self.sl_positions = self.number_of_positions
            elif self.pos_type == 1:
                if act_price < self.barrier_price:
                    self.sl = round(
                        self.zero_point - (0.1*self.mdv), self.round_number)
                    self.sl_positions = self.number_of_positions
        elif self.sl_positions != self.number_of_positions:
            if self.pos_type == 0:
                if act_price > (self.barrier_price+self.zero_point)/2:
                    self.sl = round(
                        self.zero_point + (0.1*self.mdv), self.round_number)
                    self.sl_positions = self.number_of_positions
            elif self.pos_type == 1:
                if act_price < (self.barrier_price+self.zero_point)/2:
                    self.sl = round(
                        self.zero_point - (0.1*self.mdv), self.round_number)
                    self.sl_positions = self.number_of_positions
        return act_price

    @class_errors
    def sl_giver(self):
        if self.number_of_positions >= self.symmetrical_positions:
            act_price = self.clean_sl()
            if self.pos_type == 0:
                new_sl = round(
                    act_price - self.mdv * Bot.sl_mdv_multiplier, self.round_number
                    )
                if new_sl > self.sl:
                    self.sl = new_sl
                    self.sl_positions = self.number_of_positions
            elif self.pos_type == 1:
                new_sl = round(
                    act_price + self.mdv * Bot.sl_mdv_multiplier, self.round_number
                    )
                if new_sl < self.sl:
                    self.sl = new_sl
                    self.sl_positions = self.number_of_positions
        print("Actual sl == {}".format(self.sl))

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
            "tp": self.tp,
            "sl": self.sl,
            "comment": self.comment,
            "type_time": mt.ORDER_TIME_GTC,
            "type_filling": mt.ORDER_FILLING_IOC,
            }
        order_result = mt.order_send(request)
        print(order_result)

    @class_errors
    def change_tp_sl(self):
        pos_ = [i for i in self.positions if i.comment == self.comment]
        for i in pos_:
            if (i.sl != self.sl or i.tp != self.tp) and (self.sl != 0.0 or self.tp != 0.0):
                request = {
                "action": mt.TRADE_ACTION_SLTP,
                "symbol": self.symbol,
                "position": i.ticket,
                "magic": self.magic,
                "tp": self.tp,
                "sl": self.sl,
                "comment": self.comment
                }
                order_result = mt.order_send(request)

    @class_errors
    def delete_model(self):
        os.remove(self.model_buy[1])
        print(f"Model removed: {self.model_buy[1]}")
        os.remove(self.model_sell[1])
        print(f"Model removed: {self.model_sell[1]}")

    @class_errors
    def daily_volatility_reducer(self):
        self.daily_volatility_reduce = 4
        print("New model reduce:", self.daily_volatility_reduce)

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
    def prices_list(self, posType, price_open=None):

        def price_(posType, symbol_info, i, price_open):
            if posType == 0:
                if price_open is None:
                    return round(symbol_info.bid + self.mdv * i, self.round_number)
                else:
                    return round(price_open + self.mdv * i, self.round_number)
            else:
                if price_open is None:
                    return round(symbol_info.bid + self.mdv * i, self.round_number)
                else:
                    return round(price_open + self.mdv * i, self.round_number)

        symbol_info = mt.symbol_info(self.symbol)
        if posType == 0:
            stops_range = [z for z in range(1, self.symmetrical_positions+1)]
            limits_range = [z for z in range(-2, -self.symmetrical_positions-2, -1)]
        else:
            stops_range = [z for z in [-i for i in range(1, self.symmetrical_positions+1)]]
            limits_range = [z for z in [-i for i in range(-2, -self.symmetrical_positions-2, -1)]]

        stops = [price_(posType, symbol_info, i, price_open) for i in stops_range]
        limits = [price_(posType, symbol_info, i, price_open) for i in limits_range]
        if posType == 0:
            limits.sort(reverse=True)
        else:
            limits.sort()

        print("STOPS: ", stops)
        print("LIMITS: ", limits)
        return stops, limits

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