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
#from app.model_generator import data_operations, evening_hour, probability_edge
from config.parameters import *
from app.database_class import TradingProcessor
from app.bot_functions import *
from tqdm import trange
sys.path.append("..")
mt.initialize()

print("Evening hour: ", evening_hour)

catalog = os.path.dirname(__file__)
parent_catalog = os.path.dirname(catalog)
catalog = f'{parent_catalog}\\models'
processor = TradingProcessor()


class GlobalProfitTracker:
    mt.initialize()
    def __init__(self, symbols: list, multiplier: float):
        self.barrier = round((len(symbols)*multiplier)/100, 4)
        self.global_profit_to_margin = None
        self.condition = False
        self.positions = []
        self.primal_tickets_list = []

    def checkout(self):
        account = mt.account_info()
        global_profit_to_margin = round(account.profit/account.balance, 4)
        printer("Global profit to barrier", round(global_profit_to_margin/self.barrier, 3))
        if not self.condition:
            if global_profit_to_margin > self.barrier:
                self.condition = True
                self.global_profit_to_margin = global_profit_to_margin
                self.positions = [i.ticket for i in mt.positions_get()]
                self.primal_tickets_list = self.positions
                printer("Barrier switch", self.condition)
        else:
            if global_profit_to_margin > self.global_profit_to_margin * profit_decrease_barrier:
                if global_profit_to_margin > self.global_profit_to_margin:
                    self.global_profit_to_margin = global_profit_to_margin
                pass
            else:
                if self.positions == [i.ticket for i in mt.positions_get()]:
                    close_request_("ALL", self.primal_tickets_list, True)
                    self.reset()
                else:
                    self.global_profit_to_margin = global_profit_to_margin
                    self.positions = [i.ticket for i in mt.positions_get()]

    def reset(self):
        self.global_profit_to_margin = None
        self.condition = False
        self.positions = []


class Bot:
    weekday = dt.now().weekday()
    def __init__(self, symbol):
        self.use_tracker = True if symbol == symbols[0] else False
        self.positionTracker = GlobalProfitTracker(symbols, global_tracker_multiplier) if self.use_tracker else None
        self.number_of_bars_for_backtest = 20000
        printer(dt.now(), symbol)
        self.symbol = symbol
        self.active_session()
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
        self.position_size = position_size
        self.trigger = start_trigger # 'model' 'moving_averages'
        #self.market = 'e' if dt.now().hour < change_hour else 'u'
        self.trend = 'neutral' # long_strong, long_weak, long_normal, short_strong, short_weak, short_normal, neutral
        self.trigger_model_divider = avg_daily_vol_for_divider(symbol, trigger_model_divider_factor)
        self.trend_or_not = trend_or_not(symbol)
        self.df_d1 = get_data(symbol, "D1", 1, 30)
        self.avg_daily_vol()
        self.round_number = round_number_(symbol)
        #self.mdv = self.mdv_() / 100
        self.volume_calc(position_size, True)
        self.positions_()
        #self.load_models_democracy(catalog)
        self.barOpen = mt.copy_rates_from_pos(symbol, timeframe_("M1"), 0, 1)[0][0]
        self.interval = "M1"
        self.test_strategies()

    @class_errors
    def position_time(self):
        try:
            dt_from_timestamp = dt.fromtimestamp(mt.positions_get(symbol=self.symbol)[0][1])
        except Exception:
            return 0
        return int((dt.now() - dt_from_timestamp - timedelta(hours=tz_diff)).seconds/60)


    @class_errors
    def if_tiktok(self):
        pos = mt.positions_get(symbol=self.symbol)
        profit_ = sum([pos[i].profit for i in range(len(pos)) if pos[i].magic == self.magic])
        self.close_profits.append((profit_, self.comment[:-1]))
        if len(self.close_profits) >= 2:
            x = self.close_profits[-2:]
            if all([i[0] < 0 for i in x]):
                self.position_size -= 1
                if self.position_size < 1:
                    self.position_size = 1
            elif all([i[0] > 0 for i in x]):
                self.position_size += 1
                if self.position_size > position_size + 4:
                    self.position_size = position_size + 4
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

        if self.tiktok < 1:
            if (profit_ > 0) and (last_two >= 0):
                self.tiktok -= 1
            elif (profit_ < 0) or (last_two < 0):
                self.tiktok += 1
            else:
                pass
        else:
            if (profit_ > 0) and (last_two >= 0):
                self.tiktok -= 1
            else:
                self.strategy_number += 1
                self.tiktok = 0

        if self.strategy_number > len(self.strategies)-1:
            self.test_strategies(add_number=10)
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
            interval = 'M20'
        else:
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
            print("fake_position_robot", e)
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
            #position_time = self.position_time()
            try:
                if len(self.positions) != 0:
                    profit = sum([i[-4] for i in self.positions])

                    if self.profit0 is None:
                        self.profit0 = profit
                    self.profits.append(profit+self.profit0)
                    self.profit_max = max(self.profits)
                    self.profit_min = min(self.profits)
                    mean_profits = np.mean(self.profits)
                    self.self_decline_factor()
                    if self.print_condition():
                        printer("Change value:", f"{round(self.profit_needed, 2):.2f} $")
                        #printer("Actual trigger:", self.trigger)
                        printer("Max profit:", f"{self.profit_max:.2f} $")
                        # printer("Profit zero aka spread:", f"{self.profit0:.2f} $")
                        # printer("Mean position profit minus spread:", f"{round(mean_profits-self.profit0, 2):.2f} $")
                        printer("Decline factor:", f"{self.profit_decline_factor}")
                        printer("Close position if profit is less than", f"{round(self.profit_max * self.profit_decline_factor, 2)} $")

                    if self.fake_position:
                        _ = self.fake_position_robot()

                    # Jeżeli strata mniejsza od straty granicznej
                    elif profit < -self.profit_needed*profit_decrease_barrier:# and profit > 0.91 * self.profit_min:
                        self.clean_orders()

                    # # Jeżeli strata mniejsza od straty granicznej
                    # elif profit > self.profit_needed * profit_increase_barrier and profit < profit_decrease_barrier * self.profit_max:
                    #     self.clean_orders()

                    # Jeżeli strata mniejsza od straty granicznej
                    elif self.profit_max > self.profit_needed and profit < self.profit_max * self.profit_decline_factor:
                        self.clean_orders()
                    
                    # Jeżeli zysk większy niż zysk graniczny oraz czas pozycji większy niż czas interwału oraz zysk mniejszy niż zysk maksymalny pozycji pomnożony przez współczynnik spadku
                    elif (profit > self.profit_needed/(profit_factor*1.5)):
                        _ = self.fake_position_robot()

                    if self.print_condition():
                        printer("TIKTOK:", self.tiktok)

            except Exception as e:
                print("check_trigger", e)
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
    def self_decline_factor(self, multiplier: int=2):
        min_val = 0.65
        max_val = 0.93
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
            pos_type = self.actual_position_democracy()
            self.request(actions['deal'], pos_type)
        self.positions_()

    @class_errors
    def is_this_the_end(self):
        now_ = dt.now()
        if (now_.hour == evening_hour-1 and now_.minute >= 55) or (now_.hour >= evening_hour):
            self.close_request()
            print("Na dzisiaj koniec.")
            sys.exit()
            input()

    @class_errors
    def report(self):
        time_sleep = 2
        self.pos_type = self.actual_position_democracy()
        self.positions_()
        # vvv key component vvv
        while True:
            self.is_this_the_end()
            self.request_get()
            if self.print_condition():
                printer("\nSymbol:", self.symbol)
                printer("Czas:", time.strftime("%H:%M:%S"))
            self.check_trigger(trigger_mode=trigger_mode)
            self.data()
            # track global profit
            if self.use_tracker:
                self.positionTracker.checkout()

            time.sleep(time_sleep)

    @class_errors
    def data(self, report=True):
        profit = sum([i.profit for i in self.positions if (i.magic == self.magic)])
        if self.check_new_bar():
            self.pos_type = self.actual_position_democracy()
        try:
            act_pos = self.positions[0].type
            if self.pos_type != act_pos:
                self.clean_orders()
        except (IndexError, NameError) as e:
            print("data", e)
            self.clean_orders()

        # force of strategy condition
        if self.actual_force < 1 and profit < 0:
            print("Actual strategy is weak")

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
        printer("Fake position:", self.fake_position)
        printer("Fake counter:", self.fake_counter)
        #printer("Trend:", self.trend)
        try:
            printer("Strategy name:", self.strategies[self.strategy_number][0])
        except IndexError as e:
            print("info", e)
            pass
        print()

    @class_errors
    def reset_bot(self):
        self.pos_type = None
        self.positions = None
        self.profits = []
        self.profit0 = None
        self.profit_max = 0
        # add reset fake_robot parameters
        self.fake_position = False
        self.max_close = None
        self.fake_stoploss = 0
        self.fake_counter = 0
        self.base_fake_interval = base_fake_interval
        print(f"The bot was reset.")

    @class_errors
    def avg_daily_vol(self):
        df = self.df_d1
        df['avg_daily'] = (df.high - df.low) / df.open
        self.avg_vol = df['avg_daily'].mean()

    @class_errors
    def volume_calc(self, max_pos_margin: int, min_volume: int) -> None:

        def atr():
            length = 14
            df = get_data(self.symbol, 'M5', 1, 100)
            df['atr'] = df.ta.atr(length=length)
            df['atr_osc'] = (df['atr']-df['atr'].rolling(length).min())/(df['atr'].rolling(length).max()-df['atr'].rolling(length).min()) + 0.5
            return df['atr_osc'].iloc[-1]
        try:
            another_new_volume_multiplier_from_win_rate_condition = 1 if self.win_ratio_cond else 0.6
        except AttributeError:
            another_new_volume_multiplier_from_win_rate_condition = 0.6
        max_pos_margin = int(round(max_pos_margin * atr() * another_new_volume_multiplier_from_win_rate_condition))
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
        printer('Min volume:', min_volume)
        printer('Calculated volume:', volume)
        printer("Target:", f"{self.tp_miner:.2f} $")
        printer("Killer:", f"{-self.kill_position_profit:.2f} $")

    @class_errors
    def calc_pos_condition(self, df1, window_=50):
        df = df1.copy()[-window_*10:]
        df['mkt_move'] = np.log(df.close/df.close.shift())
        df['return'] = df['mkt_move'] * df.stance.shift()
        df['strategy'] = (1+df['return']).cumprod() - 1
        df['strategy_mean'] = df['strategy'].rolling(window_).mean()
        df['strategy_std'] = df['strategy'].rolling(window_).std()/2
        df['strategy_cond'] = df['strategy_mean'] - df['strategy_std']
        df['cond'] = np.where(df['strategy']>df['strategy_cond'], 1, -2)
        df = win_ratio(df, 'return', window_)
        cond2 = df['win_ratio_fast'].iloc[-1] > df['win_ratio_slow'].iloc[-1]
        cond = df['cond'].rolling(window_).sum()
        df['date_xc'] = df['time'].dt.date
        df = df[df['date_xc'] == dt.now().date()]
        ret = (1+df['return']).cumprod() - 1
        return cond.iloc[-1], df['cond'].iloc[-1], cond2, round(ret.iloc[-1]*100, 2)

    @class_errors
    def actual_position_democracy(self, number_of_bars=250):
        try:
            if self.fake_position:
                return self.fake_position_robot()

            # market = 'e' if dt.now().hour < change_hour else 'u'
            # if market != self.market:
            #     self.market = market
            #     self.load_models_democracy(catalog)

            try:
                strategy = self.strategies[self.strategy_number]
            except IndexError as e:
                print("actual_position_democracy", e)
                self.test_strategies(add_number=10)
                strategy = self.strategies[self.strategy_number]
            print("Strategia", strategy[0])
            self.interval = strategy[0].split('_')[-1]
            fast = strategy[3]
            slow = strategy[4]
            print("Interwał", self.interval)

            # if 'model' in strategy[0]:
            #     dfx, position = self.model_position(number_of_bars, backtest=False)
            #     printer(f'Position from {strategy[0]}:', position)
            # else:
            dfx = get_data(self.symbol, self.interval, 1, int(fast * slow + 1440)) # how_many_bars
            dfx, position = strategy[1](dfx, slow, fast)
            if position not in [-1, 1]:
                dfx = get_data(self.symbol, self.interval, 1, int(fast * slow + number_of_bars*20)) # how_many_bars
                dfx, position = strategy[1](dfx, slow, fast)

                printer(f'Position from {strategy[0]}:', f'fast={fast} slow={slow}', base_just=60)
                printer(f'Position from {strategy[0]}:', position)

            self.force, self.actual_force, self.win_ratio_cond, daily_return = self.calc_pos_condition(dfx)
            self.actual_force = True if self.actual_force == 1 else False
            printer("Strategy force", self.force)
            printer("Strategy actual position", self.actual_force)
            printer("Daily return", daily_return)
            # if self.actual_force < 1:
            #     print("Next strategy, because the strategy is too weak.")

            position = int(0) if position == 1 else int(1)

            dfx['cross'] = np.where(dfx['stance'] != dfx['stance'].shift(), 1, 0)
            self.fresh_signal = True if dfx['stance'].iloc[-1] != dfx['stance'].iloc[-2] else False
            cross = dfx[dfx['cross'] == 1]
            self.strategy_pos_open_price = cross['close'].iloc[-1]

            printer("Last open position time by MetaTrader", f"{cross['time'].iloc[-1]}", base_just=60)
            if cross['time'].dt.date.iloc[-1] != dfx['time'].dt.date.iloc[-1]:
                self.strategy_number += 1
                print("Next strategy because the position is from last working day.")
                return self.actual_position_democracy()

            positions_ = mt.positions_get(symbol=self.symbol)
            if len(positions_) == 0:
                while True:
                    if self.fresh_signal:
                        break
                    self.is_this_the_end()
                    # check if price is nice to open
                    tick = mt.symbol_info_tick(self.symbol)
                    price = round((tick.ask + tick.bid) / 2, self.round_number)
                    diff = round((price - self.strategy_pos_open_price) * 100 / self.strategy_pos_open_price, 2)
                    match position:
                        case 0: self.good_price_to_open_pos = True if price <= self.strategy_pos_open_price else False
                        case 1: self.good_price_to_open_pos = True if price >= self.strategy_pos_open_price else False
                        #case 2: self.good_price_to_open_pos = True if abs(diff) < self.mdv else False
                    if self.good_price_to_open_pos:
                        break

                    if self.check_new_bar():
                        return self.actual_position_democracy(number_of_bars=number_of_bars)
                    pos = 'LONG' if position==0 else "SHORT"
                    printer('Symbol / Position / difference', f'{self.symbol} / {pos} / {diff:.2f} %', base_just=65)

                    if self.use_tracker:
                        self.positionTracker.checkout()

                    time.sleep(5)
        except KeyError as e:
            print("actual_position_democracy", e)
            return self.actual_position_democracy(number_of_bars=number_of_bars*2)
        self.pos_time = interval_time(self.interval)

        print("Pozycja", "Long" if position == 0 else "Short" if position != 0 else "None")
        return position

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

        self.volume_calc(position_size, True)
        if self.trend_or_not:
            letter = "t"
        else:
            letter = "f"

        name_ = self.strategies[self.strategy_number][0][:6]
        fast = self.strategies[self.strategy_number][3]
        slow = self.strategies[self.strategy_number][4]
        self.comment = f'{name_}_{fast}_{slow}_{self.tiktok}'

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
        # try:
        self.strategy_number += 1
        return self.actual_position_democracy()
        # except NameError as e:
        #     print(e)
        #     return random.randint(0, 1)

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
                trigger_divider=self.trigger_model_divider,
                decline_factor=self.profit_decline_factor,
                profit_factor=profit_factor,
                calculated_profit=self.profit_needed,
                minutes=self.pos_time,
                weekday=Bot.weekday,
                trend=self.trend,
                tiktok=self.tiktok,
                strategy=self.strategies[self.strategy_number][0],
                marker=self.strategies[self.strategy_number][7]
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
                fake_position=self.fake_position,
                fake_position_counter=self.fake_counter,
                fake_position_stoploss=self.fake_stoploss
            )
        except Exception as e:
            print("write_to_database", e)
            pass

    @class_errors
    def write_to_backtest(self, strategy_full_name, interval, result, kind, fast, slow):
        try:
            processor.process_backtest(
                symbol=self.symbol,
                strategy_short_name=strategy_full_name[:6],
                strategy_full_name=strategy_full_name,
                interval=interval,
                result=result,
                kind=kind,
                fast=fast,
                slow=slow
            )
        except Exception as e:
            print("write_to_backtest", e)
            pass

    @class_errors
    def trend_backtest(self, strategy):
        print(strategy.__name__)

        if strategy.__name__ == 'model':
            interval = self.model_interval
        else:
            interval = strategy.__name__.split('_')[-1]

        sharpe_multiplier = interval_time_sharpe(interval)
        df_raw = get_data(self.symbol, interval, 1, self.number_of_bars_for_backtest)
        small_bt_bars = calculate_bars_to_past(df_raw)

        #if not strategy.__name__.startswith('model'):
        results = []
        for slow in trange(5, 50):
            for fast in range(2, 21):
                if fast == slow:
                    continue
                df1, position = strategy(df_raw, slow, fast)
                df1 = calculate_strategy_returns(df1, leverage)
                df1 = delete_last_day_and_clean_returns(df1, morning_hour, evening_hour, respect_overnight)
                #df2 = df1.copy()[-small_bt_bars:]
                sharpe, calmar = calc_result(df1, sharpe_multiplier)
                sharpe2, calmar2 = calc_result(df1, sharpe_multiplier, True)
                #sharpe3, _ = calc_result(df1, sharpe_multiplier, True)
                _, actual_condition, _, daily_return = self.calc_pos_condition(df1)
                results.append((fast, slow, round(np.mean(sharpe+sharpe2), 3), np.mean(calmar+calmar2), actual_condition, daily_return))

        f_result = sorted(results, key=lambda x: x[2]*x[3], reverse=True)[0]
        print(f"Best ma factors fast={f_result[0]} slow={f_result[1]}")
        return f_result[0], f_result[1], f_result[2]*f_result[3], f_result[4], f_result[5]
        # else:
        #     df1 = self.model_position(500, backtest=True)
        #     sharpe, calmar = calc_result(df1, sharpe_multiplier)
        #     self.number_of_bars_for_backtest = small_bt_bars
        #     df2 = self.model_position(500, backtest=True)
        #     sharpe2, calmar2 = calc_result(df2, sharpe_multiplier)
        #     self.number_of_bars_for_backtest = 20000
        #     return 0, 0, round(((sharpe+sharpe2)/2)*(calmar+calmar2)/2, 3)

    @class_errors
    def sort_strategies(self):
        sorted_data = sorted(self.strategies, key=lambda x: (x[6], x[5]), reverse=True)
        first_group = sorted(self.strategies, key=lambda x: x[7], reverse=True)[0][0]
        printer("Daily starter", first_group.split('_')[-2])
        first_ = first_group.split('_')[-2]
        second_ = 'trend' if first_ == 'counter' else 'counter'
        group_t = [item for item in sorted_data if item[7] == first_]
        group_n = [item for item in sorted_data if item[7] == second_]
        alternating_data = []
        max_len = max(len(group_t), len(group_n))

        for i in range(max_len):
            if i < len(group_t):
                alternating_data.append(group_t[i])
            if i < len(group_n):
                alternating_data.append(group_n[i])
        return alternating_data

    @class_errors
    def test_strategies(self, add_number=0):
        super_start_time = time.time()
        strategies = import_strategies([])
        self.strategies_raw = []
        for strategy in strategies:
            self.is_this_the_end()
            self.check_trigger()
            name_ = strategy.__name__
            # if name_ == 'model':
            #     if self.model_counter == 0:
            #         continue
            #     interval = self.model_interval
            # else:
            interval = name_.split('_')[-1]
            kind = name_.split('_')[-2]
            #marker = "trend" if "_trend_" in name_ else "swing" if "_counter_" in name_ else "none"
            fast, slow, result, actual_condition, daily_return = self.trend_backtest(strategy)
            print(name_, interval, fast, slow, round(result, 4), actual_condition, daily_return)
            self.strategies_raw.append((name_, strategy, interval, fast, slow, round(result, 2), actual_condition, kind, daily_return))

        for name_, _, interval, fast, slow, result, _, kind, _ in self.strategies_raw:
            self.write_to_backtest(name_, interval, result, kind, fast, slow)

        self.strategies = [i for i in self.strategies_raw if ((i[5] != np.inf) and (i[5] > 0))]
        self.strategies = self.sort_strategies()

        time_info(time.time()-super_start_time, 'Total duration')

        # use only six best strategies
        if len(self.strategies) > 4+add_number:
            self.strategies = self.strategies[:4+add_number]

        if len(self.strategies) == 0:
            self.close_request()
            print("You don't have any strategy to open position right now. Waiting a half an hour for backtest.")
            sleep(1800)
            self.test_strategies()
        else:
            for i in self.strategies:
                print(i[0], i[2], i[3], i[4], i[5], i[6], i[7], i[8])
            self.strategy_number = 0


if __name__ == '__main__':
    print('Yo, wtf?')
