import pandas as pd
import pandas_ta as ta
import numpy as np
import random
import MetaTrader5 as mt
import hashlib
import time
from scipy.signal import argrelextrema
import threading
from datetime import datetime as dt
import math
import traceback
from symbols_rank import symbol_stats


def class_errors(func):
    def just_log(*args, **kwargs):
        symbol = args[0].symbol
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            time = dt.now()
            class_name = args[0].__class__.__name__
            function_name = func.__name__
            with open("class_errors.txt", "a") as log_file:
                log_file.write("Symbol {}, Time: {} Error in class {}, function {}:\n\n"
                            .format(symbol, time, class_name, function_name))
                traceback.print_exc(file=log_file)
            if isinstance(e, RecursionError ):
                print("Exit")
                input()
                exit()
            raise e
    return just_log

def timeframe_(tf):
    return getattr(mt, 'TIMEFRAME_{}'.format(tf))


def get_data(symbol, tf, start, counter):
    data = pd.DataFrame(mt.copy_rates_from_pos(
                        symbol, timeframe_(tf), start, counter))
    data["time"] = pd.to_datetime(data["time"], unit="s")
    data = data.drop(["real_volume"], axis=1)
    data.columns = ["time", "open", "high", "low",
                    "close", "volume", "spread"]
    return data


def magic_(symbol, comment):
    """
    Converts a string to an integer, using the SHA-256 hash function.
    Assigns a unique 6-digit magic number depending on the strategy name,
    symbol and interval.
    """
    expression = symbol + comment
    hash_object = hashlib.sha256(expression.encode('utf-8'))
    hash_hex = hash_object.hexdigest()
    result = int(hash_hex, 16)
    return result // 10 ** (len(str(result)) - 6)


def round_number_(symbol):
    return mt.symbol_info(symbol).digits


def real_spread(symbol):
    s = mt.symbol_info(symbol)
    return s.spread / 10**s.digits

actions = {
    # Place an order for an instant deal with the specified parameters (set a market order)
    'deal': mt.TRADE_ACTION_DEAL,
    # Place an order for performing a deal at specified conditions (pending order)
    'pending': mt.TRADE_ACTION_PENDING,
    # Change open position Stop Loss and Take Profit
    'sltp': mt.TRADE_ACTION_SLTP,
    # Change parameters of the previously placed trading order
    'modify': mt.TRADE_ACTION_MODIFY,
    # Remove previously placed pending order
    'remove': mt.TRADE_ACTION_REMOVE,
    # Close a position by an opposite one
    'close': mt.TRADE_ACTION_CLOSE_BY
    }

pendings = {
    'long_stop': mt.ORDER_TYPE_BUY_STOP,
    'short_stop': mt.ORDER_TYPE_SELL_STOP,
    'long_limit': mt.ORDER_TYPE_BUY_LIMIT,
    'short_limit': mt.ORDER_TYPE_SELL_LIMIT,
    }

class Bot:

    sl_mdv_multiplier = 2
    position_size = 1

    def __init__(self, symbol, symmetrical_positions, daily_volatility_reduce):
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
        self.pos_type = None
        self.sl = 0.0
        self.tp = 0.0
        _, self.kill_position_profit, _ = symbol_stats(self.symbol, self.volume)
        print("Killer == ", -self.kill_position_profit, " USD")

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

    def pos_creator(self):
        # return int(input("What position do You want to open? (0 -> LONG / 1 -> SHORT) "))
        return random.randint(0, 1)

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
                if act_price < (highest_price - self.mdv * Bot.sl_mdv_multiplier/2):
                    new_tp = round(highest_price + self.mdv * Bot.sl_mdv_multiplier, self.round_number)
                    self.tp = new_tp if new_tp > self.barrier_price else self.tp
            elif self.pos_type == 1:
                lowest_price = df.low.min()
                if act_price > (lowest_price + self.mdv * Bot.sl_mdv_multiplier/2):
                    new_tp = round(lowest_price - self.mdv * Bot.sl_mdv_multiplier, self.round_number)
                    self.tp = new_tp if new_tp < self.barrier_price else self.tp
            print("Actual TP == {}".format(self.tp))

    @class_errors
    def sl_giver(self):
        act_price = mt.symbol_info(self.symbol).bid
        if self.sl == 0.0:
            if self.pos_type == 0:
                if act_price > self.barrier_price:
                    self.sl = round(
                        self.zero_point + (0.1*self.mdv), self.round_number)
            elif self.pos_type == 1:
                if act_price < self.barrier_price:
                    self.sl = round(
                        self.zero_point - (0.1*self.mdv), self.round_number)
        else:
            if self.pos_type == 0:
                new_sl = round(act_price - self.mdv * Bot.sl_mdv_multiplier, self.round_number)
                self.sl = new_sl if new_sl > self.sl else self.sl
                if (act_price > self.barrier_price) and (self.sl < self.zero_point):
                    self.sl = round(
                        self.zero_point + (0.1*self.mdv), self.round_number)
            elif self.pos_type == 1:
                new_sl = round(act_price - self.mdv * Bot.sl_mdv_multiplier, self.round_number)
                self.sl = new_sl if new_sl < self.sl else self.sl
                if (act_price < self.barrier_price) and (self.sl > self.barrier_price):
                    self.sl = round(
                        self.zero_point - (0.1*self.mdv), self.round_number)
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
        if self.positions != ():
            self.positions = [i for i in self.positions if
                              (i.comment == self.comment)]
            if not len(self.positions):
                self.positions = ()

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
                def scan_orders(orders, price):
                    rel_tol = 1e-10 * 10**(8 - self.round_number)
                    math_true_false_list = [math.isclose(i.price_open, price, rel_tol=rel_tol, abs_tol=rel_tol) for i in orders]
                    print([i.price_open for i in orders])
                    print(price)
                    print("Math", math_true_false_list)
                    return any(math_true_false_list)

                if posType == 0:
                    price_ = mt.symbol_info(self.symbol).bid
                    for idx, i in enumerate(self.limits):
                        p_ = round(i+self.mdv, self.round_number)
                        if price_ < i:
                            if scan_orders(orders, p_):
                                pass
                            else:
                                self.request(actions['pending'], posType, price=p_)
                                print("self.limits przed: ", self.limits)
                                self.limits.remove(self.limits[idx])
                                print("self.limits po: ", self.limits)
                                break
                else:
                    price_ = mt.symbol_info(self.symbol).bid
                    for idx, i in enumerate(self.limits):
                        p_ = round(i-self.mdv, self.round_number)
                        if price_ > i:
                            if scan_orders(orders, p_):
                                pass
                            else:
                                self.request(actions['pending'], posType, price=p_)
                                print("self.limits przed: ", self.limits)
                                self.limits.remove(self.limits[idx])
                                print("self.limits po: ", self.limits)
                                break
                self.sl_giver()
                self.tp_giver()
                self.change_tp_sl()

            else:
                posType = self.pos_creator()
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
            price = mt.symbol_info(self.symbol).bid # mt.symbol_info(self.symbol).ask if posType == 0 else \
        else:
            if posType == 0:
                posType = pendings["long_stop"]
            else:
                posType = pendings["short_stop"]

        print(price)
        request = {
            "action": action,
            "symbol": self.symbol,
            "volume": self.volume,
            "type": posType,
            "price": price,
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
        self.number_of_positions = len(self.positions)
        try:
            self.pos_type = self.positions[0].type
        except Exception:
            print("Pozycje zamknięte")
            self.clean_orders()
            exit()
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
            print(f"RMR positions profit to positions margin:         {profit_to_margin} %")
            print(f"Percent distance from actual price to mean price: {distance} %")
            print(f"Distance to daily volatility:                     {round(distance/self.avg_vol, 2)} %")
            print(f"Actual price:                                     {act_price}")
            print(f"Barrier price:                                    {self.barrier_price}")
            print()

        if profit < -self.kill_position_profit:
            print('Loss is to high. We have to kill it!')
            self.clean_orders()

    @class_errors
    def active_session(self):
        df = get_data(self.symbol, 'D1', 0, 1)
        today_date_tuple = time.localtime()
        formatted_date = time.strftime("%Y-%m-%d", today_date_tuple)
        return str(df.time[0].date()) == formatted_date

    def close_request(self):
        positions_ = mt.positions_get(symbol=self.symbol)
        for i in positions_ :
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

            # time_sleep = int(random.randint(30, 180)*60)
            # print(f"Break {int(time_sleep/60)} minutes.")
            # time.sleep(time_sleep)
            print(f"Usunięto łącznie {counter} zleceń na symbolu {self.symbol}")
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