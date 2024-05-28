import pandas as pd
import pandas_ta as ta
import numpy as np
import random
import MetaTrader5 as mt
import time
from datetime import timedelta
# from scipy.signal import argrelextrema
# import threading
import math
from symbols_rank import symbol_stats
from functions import *

class Bot:

    sl_mdv_multiplier = 1.5 # mdv multiplier for sl
    tp_mdv_multiplier = 2   # mdv multiplier for tp
    position_size = 3       # percent of balance
    kill_multiplier = 1.5   # loss of daily volatility by one position multiplier
    tp_miner = 3

    def __init__(self, symbol, direction, symmetrical_positions, daily_volatility_reduce):
        mt.initialize()
        self.symbol = symbol
        self.symmetrical_positions = symmetrical_positions
        self.daily_volatility_reduce = daily_volatility_reduce
        self.mdv = self.MDV_() / daily_volatility_reduce
        self.volume = self.volume_calc(Bot.position_size, False)
        self.comment = f'srb_{self.symbol}_{symmetrical_positions}_{daily_volatility_reduce}'
        self.magic = magic_(self.symbol, self.comment)
        self.round_number = round_number_(self.symbol)
        self.number_of_positions = 0
        self.avg_daily_vol_()
        self.positions_()
        self.limits = None
        self.start_pos = self.pos_type = direction
        self.sl = 0.0
        self.tp = 0.0
        self.sl_positions = None
        _, self.kill_position_profit, _ = symbol_stats(self.symbol, self.volume, Bot.kill_multiplier)
        self.tp_miner = round(self.kill_position_profit * Bot.tp_miner / Bot.kill_multiplier, 2)
        print("Target == ", self.tp_miner, " USD")
        print("Killer == ", -self.kill_position_profit, " USD")

    @class_errors
    def MDV_(self):
        """Returns mean daily volatility"""
        df = get_data(self.symbol, "D1", 1, 30)
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
    def pos_creator(self):
        if (self.pos_type is None) and (self.start_pos is None):
            return random.randint(0, 1)
        else:
            return self.start_pos

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
        if self.active_session():
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
                posType = self.direction_()
                stops, _ = self.prices_list(posType)
                print(stops)
                self.request(actions['deal'], posType)
                for i in stops:
                    self.request(actions['pending'], posType, i)

            self.positions_()

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

        print(request)
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
    def report(self):
        self.positions_()
        while True:
            self.request_get()
            print("Czas:", time.strftime("%H:%M:%S"))
            self.data()
            time.sleep(5)
            print()

    @class_errors
    def data(self, report=True):
        try:
            self.pos_type = self.positions[0].type
        except Exception as e:
            print(e)
            print("Pozycje zamknięte")
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
            self.distance = 'Unknown'
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
            print()

        if profit < -self.kill_position_profit:
            print('Loss is to high. We have to kill it!')
            self.clean_orders()
        elif profit > self.tp_miner:
            print('The profit is nice. We want it on our accout.')
            self.clean_orders()

    @class_errors
    def active_session(self):
        df = get_data(self.symbol, 'D1', 0, 1)
        today_date_tuple = time.localtime()
        formatted_date = time.strftime("%Y-%m-%d", today_date_tuple)
        return str(df.time[0].date()) == formatted_date

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
    def reset_bot(self):
        self.tp = 0.0
        self.sl = 0.0
        self.limits = None
        self.pos_type = None
        self.positions = None

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
            time_sleep = int(random.randint(10, 30)*60)
            print(f"Break {int(time_sleep/60)} minutes.")
            time.sleep(time_sleep)
            self.reset_bot()
            self.report()

    @class_errors
    def avg_daily_vol_(self):
        df = get_data(self.symbol, "D1", 1, 30)
        df['avg_daily'] = (df.high - df.low) / df.open
        self.avg_vol = df['avg_daily'].mean()

    @class_errors
    def volume_calc(self, max_pos_margin, min_volume):
        symbol_info = mt.symbol_info(self.symbol)._asdict()
        if min_volume:
            return symbol_info["volume_min"]
        price = mt.symbol_info_tick(self.symbol)._asdict()
        margin_min = round(((symbol_info["volume_min"] *
                        symbol_info["trade_contract_size"])/100) *
                        price["bid"], 2)
        account = mt.account_info()._asdict()
        self.avg_daily_vol_()
        max_pos_margin = round(account["balance"] * (max_pos_margin/100) /
                            (self.avg_vol * 100))
        if "JP" not in self.symbol:
            volume = round((max_pos_margin / margin_min)) *\
                            symbol_info["volume_min"]
        else:
            volume = round((max_pos_margin * 100 / margin_min)) *\
                            symbol_info["volume_min"]
        if volume > symbol_info["volume_max"]:
            volume = float(symbol_info["volume_max"])
        return volume

    @class_errors
    def direction_(self):
        from_date = dt.today() - timedelta(days=1)
        to_date = dt.today() + timedelta(days=1)
        from_date = dt(from_date.year, from_date.month, from_date.day)
        to_date = dt(to_date.year, to_date.month, to_date.day)
        data = mt.history_deals_get(from_date, to_date)
        try:
            df_raw = pd.DataFrame(list(data), columns=data[0]._asdict().keys())
        except IndexError:
            return self.start_pos
        df_raw["time"] = pd.to_datetime(df_raw["time"], unit="s")
        df_raw = df_raw[df_raw['profit'] != 0.0]
        df = df_raw[df_raw['symbol']==self.symbol].tail(self.number_of_positions)
        check = True if 'sl' in df.comment.to_list()[0] else False
        print(df)
        if len(df) == 0:
            return int(self.start_pos)
        profit = df.profit.sum()
        type_ = df['type'].iloc[-1]
        # text = f"Same position type as last beacuse last profit = {profit}" if profit >= 0 \
        #     else f"Change positin type beacuse last profit = {profit}"
        # print(text)
        print("type: ", type_)
        # if profit >= 0:
        #     return int(0 if type_ == 1 else 1)
        # else:
        #     return int(type_)
        if check and profit < 0:
            return int(type_)
        else:
            return int(0 if type_ == 1 else 1)