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


def timeframe_(tf):
    return getattr(mt, 'TIMEFRAME_{}'.format(tf))


def actual_trend(symbol, interval, factor, x=200):
    """Get info about trend from last x bars for sepecyfied interval"""
    df = get_data(symbol, interval, 1, x)
    local_max = argrelextrema(df['high'].values,
                              np.greater, order=factor)[0]
    local_min = argrelextrema(df['low'].values,
                              np.less, order=factor)[0]
    min_max = []
    for i in local_max:
        min_max.append(df.loc[i, 'high'])
    for i in local_min:
        min_max.append(df.loc[i, 'low'])

    result = np.mean(min_max) / df.loc[0, 'close']
    trend = "LONG" if result >= 1 else "SHORT"
    print(f"Actual trend for {symbol} on {interval} interval is {trend}")
    return trend


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


class Symbol_divider:

    def __init__(self, symbols: list):
        mt.initialize()
        self.symbols = symbols
        self.comments = [f'rnd_bot_{i}' for i in symbols]
        self.symbols_in_use_()

    def symbols_in_use_(self):
        positions = mt.positions_get()
        symbols_ = [i.symbol for i in positions if i.comment in self.comments]
        self.symbols_in_use = list(set([i for i in symbols_]))
        self.unused_symbols = [i for i in self.symbols if i not in self.symbols_in_use]


class Bot:

    sl_mdv_multiplier = 0.8
    max_positions = 3
    vector = 3
    mdv_tf = "H2"
    actual_trend_tf = "H4"

    # sl_mdv_multiplier = 1
    # max_positions = 3
    # vector = 3
    # mdv_tf = "H4"
    # actual_trend_tf = "M20"

    def __init__(self, symbol, now: bool=True, limit_price: float=None):
        mt.initialize()
        self.symbol = symbol
        self.max_positions = Bot.max_positions
        self.mdv = self.MDV_() * Bot.vector / Bot.max_positions
        self.volume = mt.symbol_info(self.symbol).volume_min
        self.comment = f'rnd_bot_{self.symbol}'
        self.magic = magic_(self.symbol, self.comment)
        self.round_number = round_number_(self.symbol)
        self.number_of_positions = 0
        self.avg_daily_vol_()
        self.sl = 0.0
        self.now = now
        self.limit_price = round(limit_price, self.round_number)

    def MDV_(self):
        """Returns mean daily volatility"""
        df = get_data(self.symbol, Bot.mdv_tf, 1, 30)
        df['mean_vol'] = (df.high - df.low)
        return df['mean_vol'].mean()

    def pos_creator(self):
        trend = actual_trend(self.symbol, Bot.actual_trend_tf, 3, 200)
        trend = 0 if trend == "LONG" else 1
        df = get_data(self.symbol, Bot.actual_trend_tf, 0, 30)
        stoch = df.ta.stoch().iloc[:, 0].iloc[-1]
        rsi = df.ta.rsi().iloc[-1]
        x = 35
        pos_choice = [trend]
        if rsi > 50:
            pos_choice.append(0)
        else:
            pos_choice.append(1)

        if stoch > 50:
            pos_choice.append(0)
        else:
            pos_choice.append(1)

        position = random.choice(pos_choice)
        agreement = input(f"System propose == {position}. Are you Agree? Y/N")
        if agreement == "Y" or agreement == "y":
            return position
        else:
            return 0 if position == 1 else 1

    def tp_giver(self, posType: int, modify: bool=False):
        if modify:
            ask = bid = self.positions[0].price_open
        else:
            ask = mt.symbol_info(self.symbol).ask
            bid = mt.symbol_info(self.symbol).bid

        if posType == 0:
            price = ask
            self.tp = round(ask + (int(Bot.max_positions)) * self.mdv, self.round_number)
        else:
            price = bid
            self.tp = round(bid - (int(Bot.max_positions)) * self.mdv, self.round_number)
        return price

    def positions_(self):
        self.positions = mt.positions_get(symbol=self.symbol)
        if self.positions != ():
            self.positions = [i for i in self.positions if
                              (i.comment == self.comment)]
            if not len(self.positions):
                self.positions = ()

    def clean_orders(self):
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
            time_sleep = int(random.randint(30, 180)*60)
            print(f"Break {int(time_sleep/60)} minutes.")
            time.sleep(time_sleep)
            print(f"Usunięto łącznie {counter} zleceń na symbolu {self.symbol}")

    def sl_giver(self):
        act_price = mt.symbol_info(self.symbol).ask
        positions = [i for i in self.positions if i.comment==self.comment]
        act_sl = positions[0].sl
        print("Actual_sl: ", act_sl)
        if self.pos_type == 0:
            if act_price > self.zero_point + Bot.sl_mdv_multiplier * self.mdv:
                if act_sl == 0.0:
                    self.sl = round((self.zero_point + (1/10)*self.mdv),
                                    self.round_number)
                else:
                    df = get_data(self.symbol, 'H1', 1, 5)
                    max_ = df.high.max()
                    new_sl = round(max_ - Bot.sl_mdv_multiplier * self.mdv,
                                   self.round_number)
                    self.sl = new_sl if act_sl < new_sl else act_sl
        elif self.pos_type == 1:
            if act_price < self.zero_point - Bot.sl_mdv_multiplier * self.mdv:
                if act_sl == 0.0:
                    self.sl = round((self.zero_point - (1/10)*self.mdv),
                                    self.round_number)
                else:
                    df = get_data(self.symbol, 'H1', 1, 5)
                    min_ = df.low.min()
                    new_sl = round(min_ + Bot.sl_mdv_multiplier * self.mdv,
                                   self.round_number)
                    self.sl = new_sl if act_sl > new_sl else act_sl
        print("self.sl: ", self.sl)

    def one_order(self, price, posType, action):
        for _ in range(1, int(Bot.max_positions/3)+1):
            if posType == 0:
                pendingType = mt.ORDER_TYPE_BUY_LIMIT
                price = round(price-(((self.mdv*2))), self.round_number)
                self.request(action, pendingType, price)
            elif posType == 1:
                pendingType = mt.ORDER_TYPE_SELL_LIMIT
                price = round(price+(self.mdv*2), self.round_number)
                self.request(action, pendingType, price)

    def now_gate(self, posType):
        if self.now:
            pass
        else:
            if posType == 0:
                while mt.symbol_info(self.symbol).ask > self.limit_price:
                    time.sleep(1)
                    if dt.now().second == 30:
                        poz = "LONG"
                        print(f"Waiting for {poz} position.")
                        print("Price is {}. Waiting when limit point {} will be reached."
                              .format(mt.symbol_info(self.symbol).ask, self.limit_price))
            elif posType == 1:
                while mt.symbol_info(self.symbol).bid < self.limit_price:
                    time.sleep(1)
                    if dt.now().second == 30:
                        poz = "SHORT"
                        print(f"Waiting for {poz} position.")
                        print("Price is {}. Waiting when limit point {} will be reached."
                              .format(mt.symbol_info(self.symbol).ask, self.limit_price))

    def request_get(self):
        if self.active_session():
            self.positions_()
            if self.positions:
                print(self.positions)
            if self.positions == () or self.positions == []:
                posType = self.pos_creator()
                self.now_gate(posType)
                action = mt.TRADE_ACTION_DEAL
                price = self.tp_giver(posType)

                print("TYPE: ", posType,
                "VOLUME: ", self.volume,
                "PRICE: ", price,
                "MDV: ", self.mdv)

                self.clean_orders()
                self.request(action, posType, price)
                self.positions_()
                if mt.orders_get(symbol=self.symbol) == ():
                    action = mt.TRADE_ACTION_PENDING
                    self.one_order(price, posType, action)
                    for _ in range(1, self.max_positions):
                        if posType == 0:
                            pendingType = mt.ORDER_TYPE_BUY_STOP
                            price = round(price+self.mdv, self.round_number)
                            print("PRICE: ", price)
                            self.request(action, pendingType, price)
                        if posType == 1:
                            pendingType = mt.ORDER_TYPE_SELL_STOP
                            price = round(price-self.mdv, self.round_number)
                            print("PRICE: ", price)
                            self.request(action, pendingType, price)
                self.positions_()
            else:
                self.data(report=False)
                pt = self.pos_type
                _ = self.tp_giver(pt, modify=True)
                self.sl_giver()
                print("SELF POSITIONS TP", self.positions[0].tp)
                print("SELF TP: ", self.tp)
                print("SELF POSITIONS SL", self.positions[0].sl)
                print("SELF SL: ", self.sl)
                if self.positions[0].tp == self.tp and self.tp != 0:
                    pass

                else:
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
                        # print(f"TP {self.symbol} CHANGED")
                        print()

        else:
            print(f"Session on {self.symbol} is unactive!")
            pass

    def request(self, action, posType, price):
        request = {
            "action": action,
            "symbol": self.symbol,
            "volume": self.volume,
            "type": posType,
            "price": price,
            "deviation": 10,
            "magic": self.magic,
            "tp": self.tp,
            "comment": self.comment,
            "type_time": mt.ORDER_TIME_GTC,
            "type_filling": mt.ORDER_FILLING_IOC,
            }
        order_result = mt.order_send(request)
        print(order_result)

    def hedge_request(self, action, posType, price, tp, sl):
        request = {
            "action": action,
            "symbol": self.symbol,
            "volume": self.volume,
            "type": posType,
            "price": price,
            "deviation": 10,
            "magic": self.magic,
            "tp": tp,
            "sl": sl,
            "comment": self.comment+'_h',
            "type_time": mt.ORDER_TIME_GTC,
            "type_filling": mt.ORDER_FILLING_IOC,
            }
        order_result = mt.order_send(request)
        print(order_result)

    def avg_daily_vol_(self):
        df = get_data(self.symbol, "D1", 1, 30)
        df['avg_daily'] = (df.high - df.low) / df.open
        self.avg_vol = df['avg_daily'].mean()

    def report(self):
        self.positions_()
        _ = actual_trend(self.symbol, 'H1', 3)
        while True:
            self.request_get()
            print("Czas:", time.strftime("%H:%M:%S"))
            self.data()
            time.sleep(20)
            print()

    def data(self, report=True):
        self.number_of_positions = len(self.positions)
        self.pos_type = self.positions[0].type
        account = mt.account_info()
        act_price = mt.symbol_info(self.symbol).ask
        profit = sum([i.profit for i in self.positions if
                      ((i.comment == self.comment) and i.magic == self.magic)])
        mean_open_price = np.mean([i.price_open for i in self.positions if
                      ((i.comment == self.comment) and i.magic == self.magic)])
        spread = real_spread(self.symbol) #*self.number_of_positions*2

        profit_to_margin = round((profit/account.margin)*100, 2)

        if self.pos_type == 0:
            distance = round(((act_price - mean_open_price) / mean_open_price)*100, 2)
            self.zero_point = round(mean_open_price + spread, self.round_number)
            barrier_price = round(self.zero_point + Bot.sl_mdv_multiplier * self.mdv, self.round_number)
        elif self.pos_type == 1:
            distance = round(((mean_open_price - act_price) / mean_open_price)*100, 2)
            self.zero_point = round(mean_open_price - spread, self.round_number)
            barrier_price = round(self.zero_point - Bot.sl_mdv_multiplier * self.mdv, self.round_number)
        else:
            self.distance = 'Unknown'
            self.zero_point = 'Unknown'
            barrier_price = 'Unknown'


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
            print(f"Barrier price:                                    {barrier_price}")
            print()

    def active_session(self):
        df = get_data(self.symbol, 'D1', 0, 1)
        today_date_tuple = time.localtime()
        formatted_date = time.strftime("%Y-%m-%d", today_date_tuple)
        return str(df.time[0].date()) == formatted_date


class Control(Symbol_divider):

    def __init__(self, symbols: list):
        super().__init__(symbols)

    def globals_i(self, i):
        globals()[f'instance_{i}'] = Bot(i)
        globals()[f'thread_{i}'] = threading.Thread(
            target=globals()[f'instance_{i}'].report)

    def control_panel(self):
        if self.unused_symbols:
            print("UNUSED_SYMBOLS")
            for i in self.unused_symbols:
                print(f'Symbol {i} not in use:')
                if int(input(f"Do You want to start random mean system for {i}? 1/0 ")):
                    if int(input(f"Are You sure? 1/0 ")):
                        self.globals_i(i)

        for i in self.symbols_in_use:
            print("SYMBOLS_IN_USE")
            self.globals_i(i)

        for n in self.symbols:
            for i in globals():
                if f"thread_{n}" in i:
                    globals()[i].start()
                    time.sleep(5)

        print("")


