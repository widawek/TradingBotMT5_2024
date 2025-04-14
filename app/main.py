import pandas as pd
import numpy as np
import MetaTrader5 as mt
import time
import sys
import os
from datetime import timedelta, timezone
from datetime import datetime as dt
from time import sleep
from extensions.symbols_rank import symbol_stats, tp_sl_in_currency
from app.functions import *
from app.decorators import class_errors, measure_time
from config.parameters import *
from app.database_class import TradingProcessor
from strategies_analize.global_strategies import *
from strategies_analize.metrics import *
from app.bot_functions import *
import json
from app.montecarlo import Montecarlo, PermutatedDataFrames
from tqdm import trange
from app.minor_classes import Reverse, Target, GlobalProfitTracker
sys.path.append("..")
mt.initialize()

print("Evening hour: ", evening_hour)

catalog = os.path.dirname(__file__)
parent_catalog = os.path.dirname(catalog)
catalog = f'{parent_catalog}\\models'
processor = TradingProcessor()
time_diff = get_timezone_difference()


class Bot:
    montecarlo_for_all = True
    target_class = Target()
    weekday = dt.now().weekday()
    def __init__(self, symbol):
        self.position_capacity = []
        self.backtest_time = dt.now()
        self.reverse = Reverse(symbol)
        self.if_position_with_trend = 'n'
        self.fresh_daily_target = False
        self.currency = mt.account_info().currency
        self.pwt_short, self.pwt_long, self.pwt_dev, self.pwt_divider = play_with_trend_bt(symbol)
        self.after_change_hour = False if dt.now().hour < change_hour else True
        self.actual_today_best = 'x'
        self.use_tracker = True if symbol == symbols[0] else False
        self.positionTracker = GlobalProfitTracker(symbols, global_tracker_multiplier) if self.use_tracker else None
        self.number_of_bars_for_backtest = 40000
        printer(dt.now(), symbol)
        self.symbol = symbol
        #self.active_session()
        self.magic = magic_(symbol, 'bot_2025')
        self.profit0 = None
        self.fresh_signal = None
        self.strategy_pos_open_price = None
        self.good_price_to_open_pos = None
        self.print_count = 0
        self.change = 0
        self.tiktok = 0
        self.number_of_positions = 0
        self.profit_max = 0
        self.profits = []
        self.close_profits = []
        self.global_positions_stats = []
        self.position_size = position_size
        self.trend = 'neutral'
        self.trigger_model_divider = avg_daily_vol_for_divider(symbol, trigger_model_divider_factor)
        self.trend_or_not = trend_or_not(symbol)
        self.df_d1 = get_data(symbol, "D1", 1, 40)
        self.avg_daily_vol()
        self.round_number = round_number_(symbol)
        self.volume_calc(position_size, 0, True)
        self.positions_()
        self.barOpen = mt.copy_rates_from_pos(symbol, timeframe_("M1"), 0, 1)[0][0]
        self.interval = "M1"
        self.drift = "Neutral"
        self.virgin_test = True
        self.request_sltp = True
        self.test_strategies()

    @class_errors
    def position_time(self):
        try:
            dt_from_timestamp = dt.fromtimestamp(mt.positions_get(symbol=self.symbol)[0][1])
        except Exception:
            return 0
        return round((dt.now() - dt_from_timestamp - timedelta(hours=tz_diff)).seconds/60)

    @class_errors
    def drift_giver(self):
        x = self.close_profits[-2:]
        last_profit = self.close_profits[-1][0]

        # "Neutral" "Strong loss" "Weak loss" "Strong win" "Weak win"
        match self.drift:
            case "Neutral": decline, increase = 0.9, 1.11
            case "Strong loss": decline, increase = 1.43, 0.7
            case "Weak loss": decline, increase = 0.8, 1.25
            case "Strong win": decline, increase = 0.7, 1.43
            case "Weak win": decline, increase = 1.25, 0.8

        if last_profit < 0 and x[0][1][:6] == x[1][1][:6]:
            self.position_size = decline*self.position_size
        elif last_profit > 0 and x[0][1][:6] == x[1][1][:6]:
            self.position_size = increase*self.position_size
        else:
            pass

    @class_errors
    def tiktok_slcheck(self):
        try:
            tt=0
            positions = mt.positions_get(symbol=self.symbol)
            try:
                tt = positions[0].ticket
                print(tt)
            except Exception:
                print('no position')
            dzisiaj = dt.now().date()
            poczatek_dnia = dt.combine(dzisiaj, dt.min.time())
            koniec_dnia = dt.now() + timedelta(days=2)
            zamkniete_transakcje = mt.history_deals_get(poczatek_dnia, koniec_dnia, group=self.symbol)
            zamkniete_transakcje = [i for i in zamkniete_transakcje if i.position_id != tt]
            if len(zamkniete_transakcje) != 0:
                comment = zamkniete_transakcje[-1].comment
                profit = zamkniete_transakcje[-1].profit
                print(comment, profit)
                return 'sl' in comment, profit
            return False, 0
        except Exception as e:
            print("tiktok_slcheck", e)
            return False, 0

    @class_errors
    def if_tiktok(self, backtest=False):
        last_pos_by_sl, last_profit = self.tiktok_slcheck()

        try:
            self.drift = self.strategies[self.strategy_number][12]
        except Exception:
            self.drift = self.drift

        pos = mt.positions_get(symbol=self.symbol)
        profit_ = sum([pos[i].profit for i in range(len(pos)) if pos[i].magic == self.magic])
        self.close_profits.append((profit_, self.comment[:-1]))
        if len(self.close_profits) >= 2:
            self.drift_giver()
            try:
                last_to_by_comment = [i[0] for i in profit_ if i[1] == self.comment[:-1]]
                if len(last_to_by_comment) >= 2:
                    last_two = sum(last_to_by_comment[-2:])
                    printer("Last two positions profit", f"{last_two:.2f} {self.currency}")
                else:
                    last_two = 0
            except Exception:
                last_two = 0
        else:
            last_two = 0

        #if '_0_0_' not in self.comment:
        if self.tiktok < 1:
            if ((profit_ > 0) and (last_two >= 0)) or (last_pos_by_sl and last_profit > 0):
                self.tiktok -= 1
            elif (profit_ < 0) or (last_pos_by_sl and last_profit < 0):
                self.tiktok += 1
            else:
                pass
        else:
            if ((profit_ > 0) and (last_two >= 0)) or (last_pos_by_sl and last_profit > 0):
                self.tiktok -= 1
            else:
                self.new_strategy()
                self.tiktok = 0
                self.position_size = position_size
        # if not backtest:
        #     if self.strategy_number > len(self.strategies)-1:
        #         self.test_strategies()
        self.tiktok = 0 if self.tiktok < 0 else self.tiktok

    @class_errors
    def check_trigger(self, backtest=False):
        try:
            if self.positions is None or len(self.positions) != 0:
                profit = sum([i[-4] for i in self.positions])
                multi = 1
                x, y = Bot.target_class.checkTarget()
                if x:
                    multi = 2
                if y:
                    self.fresh_daily_target = True
                if self.profit0 is None:
                    self.profit0 = profit
                self.profits.append(profit+self.profit0)
                self.profit_max = max(self.profits)
                self.profit_min = min(self.profits)
                self.self_decline_factor()

                kind_ = self.strategies[self.strategy_number][7]
                sl = min([self.profit_needed, self.sl_money])
                tp = min([round(self.profit_needed*profit_increase_barrier*2, 2), self.tp_money])
                # if kind_ == 'counter':
                #     sl = round(sl/1.1, 2)

                if self.print_condition():
                    printer("Kind:", kind_)
                    printer("Change value:", f"{round(self.profit_needed, 2):.2f} ({self.profit_needed_min:.2f}) {self.currency}")
                    printer("TP now:", f"{round(tp, 2):.2f} {self.currency}")
                    printer("SL now:", f"{round(sl, 2):.2f} {self.currency}")
                    printer("TP from strategy:", f"{round(self.tp_money, 2):.2f} {self.currency}")
                    printer("SL from strategy:", f"{round(self.sl_money, 2):.2f} {self.currency}")
                    printer("Max profit:", f"{self.profit_max:.2f} {self.currency}")
                    printer("Decline factor:", f"{self.profit_decline_factor}")
                    printer("Close position if profit is less than", f"{round(self.profit_max * self.profit_decline_factor, 2)} {self.currency}")

                # Jeżeli strata mniejsza od straty granicznej
                if profit < -sl/self.too_much_risk():# and profit > 0.91 * self.profit_min:
                    self.clean_orders(backtest)

                # Jeżeli strata mniejsza od straty granicznej
                elif ((self.profit_max > tp/multi and profit < self.profit_max * self.profit_decline_factor) or
                    (self.fresh_daily_target and profit < self.profit_max * (self.profit_decline_factor-0.06))):
                    self.clean_orders(backtest)

                else:
                    if self.check_capacity():
                        self.change_tp_sl(tp, True)
                    self.change_tp_sl(tp)

                if self.print_condition():
                    printer("TIKTOK:", self.tiktok)

        except Exception as e:
            print("check_trigger", e)
            pass

    @class_errors
    def clean_orders(self, backtest=False):
        self.if_tiktok(backtest)
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
        if len(self.positions) != 0 and isinstance(self.positions, tuple):
            self.positions = [i for i in self.positions if (i.magic == self.magic)]

    @class_errors
    def request_get(self):
        self.positions_()
        if not self.positions:
            self.reset_bot()
            self.request(actions['deal'], self.actual_position_democracy())
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
        self.request_get()
        # vvv key component vvv
        while True:
            self.is_this_the_end()
            self.request_get()
            self.open_pos_capacity()
            if self.print_condition():
                printer("\nSymbol:", self.symbol)
                printer("Czas:", time.strftime("%H:%M:%S"))
            self.check_trigger()
            self.data()
            # track global profit
            if self.use_tracker:
                self.positionTracker.checkout()
            time.sleep(time_sleep)

    @class_errors
    def data(self, report=True):
        profit = sum([i.profit for i in self.positions if (i.magic == self.magic)])
        if self.check_new_bar() or self.pos_type is None:
            self.pos_type = self.actual_position_democracy()
        try:
            act_pos = self.positions[0].type
            if int(self.pos_type) != int(act_pos):
                self.clean_orders()
        except Exception as e:
            print("data", e)
            self.clean_orders()

        self.force_profit_print = 'ok'
        if self.actual_force < 1 and profit < 0:
            self.force_profit_print = 'weak'

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
        printer("Profit:", f"{round(profit, 2):.2f} {self.currency}")
        printer("Account balance:", f"{account.balance:.2f} {self.currency}")
        printer("Account free margin:", f"{account.margin_free:.2f} {self.currency}")
        printer("Profit to margin:", f"{profit_to_margin:.2f} %")
        printer("Profit to balance:", f"{profit_to_balance:.2f} %")
        printer("Actual position from model:", self.pos_type)
        try:
            printer("Strategy name:", self.strategies[self.strategy_number][0])
            printer("Last backtest:", self.backtest_time)
            printer("Actual strategy is",  f"{self.force_profit_print}")
        except IndexError as e:
            print("info", e)
            pass
        print()

    @class_errors
    def reset_bot(self):
        self.pos_type = None
        self.positions = ()
        self.profits = []
        self.profit0 = None
        self.profit_max = 0
        self.fresh_daily_target = False
        self.position_capacity = []

    @class_errors
    def avg_daily_vol(self):
        self.avg_vol = ((self.df_d1.high - self.df_d1.low) / self.df_d1.open).mean()

    @class_errors
    def volume_calc(self, max_pos_margin: float, posType: int, min_volume: bool) -> None:

        def atr():
            df = get_data(self.symbol, 'M5', 1, 20*12*24)
            df['hour'] = df['time'].dt.hour
            daily = df.iloc[-50:].copy()
            df = df[(df['hour'] > 9)&(df['hour'] < 20)]
            volatility_5 = ((df['high']-df['low'])/df['open']).mean()
            volatility_d = ((daily['high']-daily['low'])/daily['open']).mean()
            return round((volatility_d/volatility_5), 4)

        try:
            another_new_volume_multiplier_from_win_rate_condition = 1 if self.win_ratio_cond else 0.6
        except AttributeError:
            another_new_volume_multiplier_from_win_rate_condition = 0.6

        bonus = play_with_trend(self.symbol, self.pwt_short, self.pwt_long, self.pwt_dev, self.pwt_divider)

        # antitrend = 1
        try:
            strategy = self.strategies[self.strategy_number]
        except Exception as e:
            print('volume_calc anti trend no strategy', e)

        trend_bonus = bonus if posType == 0 else -bonus
        volume_m15 = self.volume_reducer(posType, 'M15')
        volume_m20 = self.volume_reducer(posType, 'M20')
        if volume_m15 == 1 and volume_m20 == 1:
            self.if_position_with_trend = 'y'
        elif volume_m15 != 1 and volume_m20 != 1:
            self.if_position_with_trend = 'n'
        elif volume_m15 == 1 and volume_m20 != 1:
            self.if_position_with_trend = 's'
        elif volume_m15 != 1 and volume_m20 == 1:
            self.if_position_with_trend = 'l'

        max_pos_margin2 = max_pos_margin * atr() * another_new_volume_multiplier_from_win_rate_condition
        max_pos_margin2 = (max_pos_margin2 + max_pos_margin2*trend_bonus)*volume_m15*volume_m20
        try:
            max_pos_margin2 = max_pos_margin2 / vol_cond_result(strategy[14], posType)
        except Exception as e:
            print("volume_condition: ", e)
            pass
        x, _ = Bot.target_class.checkTarget()

        if x:
            max_pos_margin2 = max_pos_margin2 / 2
        print('max_pos_margin', round(max_pos_margin2, 3))

        info_ = mt.account_info()
        if (info_.margin_free < info_.balance/10) and (not x):
            max_pos_margin2 = max_pos_margin2 / 5

        leverage_ = info_.leverage
        symbol_info = mt.symbol_info(self.symbol)._asdict()
        price = mt.symbol_info_tick(self.symbol)._asdict()
        margin_min = round(((symbol_info["volume_min"] *
                        symbol_info["trade_contract_size"])/leverage_) *
                        price["bid"], 2)
        account = info_._asdict()
        max_pos_margin2 = round(account["balance"] * (max_pos_margin2/100) /
                            (self.avg_vol * 100))
        divider_condition = 1 if self.too_much_risk() == 1 else 2
        if "JP" not in self.symbol:
            volume = round((max_pos_margin2 / (margin_min*divider_condition))) *\
                            symbol_info["volume_min"]
            printer('Volume from value:', round((max_pos_margin2 / margin_min), 2))
        else:
            volume = round((max_pos_margin2 * 100 / (margin_min*divider_condition))) *\
                            symbol_info["volume_min"]
            printer('Volume from value:', round((max_pos_margin2 * 100 / margin_min), 2))
        if volume > symbol_info["volume_max"]:
            volume = float(symbol_info["volume_max"])
        self.volume = volume
        if min_volume or (volume < symbol_info["volume_min"]):
            self.volume = symbol_info["volume_min"]
        _, self.kill_position_profit, _ = symbol_stats(self.symbol, self.volume, kill_multiplier)
        self.kill_position_profit = round(self.kill_position_profit, 2)# * (1+self.multi_voltage('M5', 33)), 2)
        self.tp_miner = round(self.kill_position_profit * tp_miner / kill_multiplier, 2)
        self.profit_needed = round(self.kill_position_profit/self.trigger_model_divider, 2)
        self.profit_needed_min = round(self.profit_needed / (self.volume/symbol_info["volume_min"]), 2)
        self.fresh_daily_target = False
        printer('Min volume:', min_volume)
        printer('Calculated volume:', volume)
        printer("Target:", f"{self.tp_miner:.2f} {self.currency}")
        printer("Killer:", f"{-self.kill_position_profit:.2f} {self.currency}")

        try:
            tp_percent = strategy[10]
            sl_percent = strategy[11]
            self.tp_money, self.sl_money = tp_sl_in_currency(self.symbol, self.volume, tp_percent, sl_percent)
            printer("Calculated takeprofit:", f"{self.tp_money:.2f} {self.currency}")
            printer("Calculated stoploss:", f"{self.sl_money:.2f} {self.currency}")
        except Exception as e:
            print('volume_calc anti trend no strategy', e)

    @class_errors
    def calc_pos_condition(self, df1, window_=50):
        df = df1.copy()[-window_*30:]
        df, _ = calculate_strategy_returns(df, leverage)
        df['strategy'] = (1+df['return']).cumprod() - 1
        df['strategy_mean'] = df['strategy'].rolling(window_).mean()
        df['strategy_std'] = df['strategy'].rolling(window_).std()/2
        df['strategy_cond'] = df['strategy_mean'] - df['strategy_std']
        df['cond'] = np.where(df['strategy']>df['strategy_cond'], 1, -2)
        df = win_ratio(df, 'return', window_)
        cond2 = df['win_ratio_fast'].iloc[-1] > df['win_ratio_slow'].iloc[-1]
        cond = df['cond'].rolling(window_).sum()
        df['date_xy'] = df['time'].dt.date
        df = df[df['date_xy'] == dt.now().date()]
        last_index = df[df['cross'] == 1].last_valid_index()
        df = df.loc[:last_index]
        if not respect_overnight:
            df['return'] = np.where((df['time'].dt.hour < morning_hour-1) | (df['time'].dt.hour > evening_hour+1), np.NaN, df['return'])
            df = df.dropna()
        # ret = (1+df['return']).cumprod() - 1
        # ret = df['return'].mean()/df['return'].std()
        # ret = round(ret, 4)
        try:
            sharpe, omega = calc_result(df, 1, False, False)
            ret = round(sharpe * omega, 8)
            return cond.iloc[-1], df['cond'].iloc[-1], cond2, ret
        except IndexError:
            return 0, 0, 0, 0

    @class_errors
    def actual_position_democracy(self, number_of_bars=250):
        try:
            try:
                strategy = self.strategies[self.strategy_number]
            except IndexError as e:
                print("actual_position_democracy", e)
                #self.number_of_bars_for_backtest = int(self.number_of_bars_for_backtest/2)
                self.test_strategies()
                strategy = self.strategies[self.strategy_number]
            print("Strategia", strategy[0])
            self.interval = strategy[2]
            fast = strategy[3]
            slow = strategy[4]

            dfx, stance = strategy[1](get_data(self.symbol, self.interval, 1, int(fast * slow + 500)), slow, fast, self.symbol)
            if stance not in [-1, 1]:
                dfx = get_data(self.symbol, self.interval, 1, int(fast * slow + number_of_bars*20)) # how_many_bars
                dfx, stance = strategy[1](dfx, slow, fast, self.symbol)

            printer(f'Position from {strategy[0]}:', f'fast={fast} slow={slow}', base_just=60)
            printer(f'Position from {strategy[0]}:', stance)

            self.force, self.actual_force, self.win_ratio_cond, daily_return = self.calc_pos_condition(dfx)
            self.actual_force = True if self.actual_force == 1 else False

            # if strategy[15] == -1:# and self.backtest_time.hour < 12:
            #     print("Position is reverse by backtest weekday results.")
            #     stance = int(stance*strategy[15])

            position = int(0) if stance == 1 else int(1)

            # if self.reverse.reverse_or_not():
            #     mode__ = "REVERSE"
            #     print("REVERSE MODE")
            #     position = int(0) if position == 1 else int(1)
            # else:
            #     mode__ = "NORMAL"
            #     print("NORMAL MODE")

            mode__ = "NORMAL"

            # # everything reverse test
            # position = int(0) if stance == -1 else int(1)

            dfx['cross'] = np.where(dfx['stance'] != dfx['stance'].shift(), 1, 0)
            self.fresh_signal = True if dfx['stance'].iloc[-1] != dfx['stance'].iloc[-2] else False
            cross = dfx[dfx['cross'] == 1]
            self.strategy_pos_open_price = cross['close'].iloc[-1]
            printer("Last open position time by MetaTrader", f"{cross['time'].iloc[-1]}", base_just=60)
            # if dt.now().hour > 12:
            #     if cross['time'].dt.date.iloc[-1] != dfx['time'].dt.date.iloc[-1]:
            #         self.strategy_number += 1
            #         print("Next strategy because the position is from last working day.")
            #         return self.actual_position_democracy()

            positions_ = mt.positions_get(symbol=self.symbol)
            #kind = strategy[0].split('_')[1]
            if len(positions_) == 0:
                minute_ = 0
                minutes = 0
                while True:
                    if self.fresh_signal and minutes > interval_time(self.interval):
                        break
                    self.is_this_the_end()
                    # check if price is nice to open
                    tick = mt.symbol_info_tick(self.symbol)
                    price = round((tick.ask + tick.bid) / 2, self.round_number)
                    diff = round((price - self.strategy_pos_open_price) * 100 / self.strategy_pos_open_price, 2)

                    match position:
                        case 0: self.good_price_to_open_pos = True if rsi_condition(self.symbol, 0, self.interval) else False #(price <= self.strategy_pos_open_price) and rsi_condition(self.symbol, 0) else False
                        case 1: self.good_price_to_open_pos = True if rsi_condition(self.symbol, 1, self.interval) else False #(price >= self.strategy_pos_open_price) and rsi_condition(self.symbol, 0) else False

                    if self.good_price_to_open_pos:
                        break

                    if self.check_new_bar():
                        return self.actual_position_democracy(number_of_bars=number_of_bars*10)
                    pos = 'LONG' if position==0 else "SHORT"
                    new_minute = dt.now().minute
                    if minute_ != new_minute:
                        printer('Symbol / Position / difference', f'{self.symbol} / {pos} / {diff:.2f} %', base_just=65)
                        minute_ = new_minute
                        minutes += 1

                    if self.use_tracker:
                        self.positionTracker.checkout()

                    time.sleep(5)
        except KeyError as e:
            try:
                print("actual_position_democracy", e)
                self.test_strategies()
                return self.actual_position_democracy(number_of_bars=number_of_bars*2)
            except Exception as e:
                print(e)
                return self.pos_type
        self.pos_time = interval_time(self.interval)

        printer("Daily return", daily_return)
        printer("POZYCJA", "LONG" if position == 0 else "SHORT" if position != 0 else "None" + f"w trybie {mode__}")
        return position

    @class_errors
    def last_pos_sltp(self):
        try:
            tt=0
            positions = mt.positions_get(symbol=self.symbol)
            try:
                tt = positions[0].ticket
                print(tt)
            except Exception:
                print('no position')
            dzisiaj = dt.now().date()
            poczatek_dnia = dt.combine(dzisiaj, dt.min.time())
            koniec_dnia = dt.now() + timedelta(days=2)
            zamkniete_transakcje = mt.history_deals_get(poczatek_dnia, koniec_dnia, group=self.symbol)
            zamkniete_transakcje = [i for i in zamkniete_transakcje if i.position_id != tt]
            intervals = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12', 'M15', 'M20', 'M30', 'H1']
            rsi_interval = int(intervals[intervals.index(self.interval)+6][1:])
            comment = zamkniete_transakcje[-1].comment
            condition = ('tp' in comment) or ('sl' in comment)
            return condition, rsi_interval
        except Exception as e:
            print("last_pos_sltp", e)
            return False, 30

    @class_errors
    def request(self, action, posType, price=None):
        cond, time_ = self.last_pos_sltp()
        if cond and self.request_sltp:
            self.request_sltp = False
            print(f"Sleep {time_} minutes after sl or tp")
            time.sleep(time_*60)
            return self.request_get()

        if action == actions['deal']:
            print("YES")
            price = mt.symbol_info(self.symbol).bid
        else:
            if posType == 0:
                posType = pendings["long_stop"]
            else:
                posType = pendings["short_stop"]

        self.volume_calc(self.position_size, posType, False)

        strategy = self.strategies[self.strategy_number]
        name_ = strategy[0][:5]
        fast = strategy[3]
        slow = strategy[4]
        reverseornot = 'n' if strategy[15] != -1 else 'r'

        metric_numb = str(metric_numb_dict[self.bt_metric.__name__])
        self.comment = f'{name_}{self.interval[-1:]}_{fast}_{slow}_{reverseornot}{metric_numb}{self.if_position_with_trend}'

        # if self.reverse.condition:
        #     self.comment = '8'+self.comment[1:]

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
        if order_result.comment == 'No money':
            seconds = 300
            print(f"No enough money to open position. Waiting {round(seconds/60, 2)} minutes")
            sleep(300)
            self.request_get()
        self.request_sltp = True
        self.position_capacity = []

    @class_errors
    def open_pos_capacity(self):
        if len(self.position_capacity) == 0:
            positions = mt.positions_get(symbol=self.symbol)
            positions = [i for i in positions if i.magic == self.magic]
            type_ = positions[0].type
            oprice = positions[0].price_open
            cprice = positions[0].price_current
            self.first_price_diff = abs(cprice - oprice)

        try:
            position = mt.positions_get(symbol=self.symbol)[0]
            if position.magic != self.magic:
                return
            type_ = position.type
            oprice = position.price_open
            cprice = position.price_current
            drop = 0
            if type_ == 0:
                drop = (cprice - oprice) + self.first_price_diff
            elif type_ == 1:
                drop = (oprice - cprice) + self.first_price_diff
            self.position_capacity.append(drop)
        except Exception as e:
            print('open_pos_capacity', e)
            return

    @class_errors
    def check_capacity(self):
        try:
            position_efficiency = len([i for i in self.position_capacity if i > 0]) / len(self.position_capacity)
            capacity = np.mean(self.position_capacity)
            efficiency = position_efficiency
            print("Position capacity:  ", round(capacity, 5))
            print("Position efficiency:", round(efficiency, 3))
            if capacity < 0 and efficiency < 0.49 and self.duration():
                return True
            return False
        except Exception:
            return False

    @class_errors
    def duration(self):
        intervals = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12', 'M15', 'M20', 'M30', 'H1']
        duration_time = int(intervals[intervals.index(self.interval)+6][1:])
        return self.position_time() > duration_time

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
        return df['mean_vol'].mean() + df["mean_vol"].std()

    @class_errors
    def active_session(self):
        df = get_data(self.symbol, 'H1', 0, 1)
        today_date_tuple = time.localtime()
        formatted_date = time.strftime("%Y-%m-%d", today_date_tuple)
        if str(df.time.dt.date.iloc[-1]) == formatted_date:
            pass
        else:
            print(f"Session on {self.symbol} is not active")
            input()
            sys.exit(1)

    @class_errors
    def close_request(self, in_=False):
        positions_ = mt.positions_get(symbol=self.symbol)
        if positions_:
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
        else:
            print("Any position was opened.")

        time_from_last_backtest_hours = round((dt.now() - self.backtest_time).seconds/3600, 3)
        printer('time_from_last_backtest_hours', time_from_last_backtest_hours)

        hours_ = 8
        if time_from_last_backtest_hours >= hours_ and not in_:
            print(f"Last backtest was {hours_} hour ago. I need new data.")
            self.test_strategies()


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
                spread=spread
            )
        except Exception as e:
            print("write_to_database", e)
            pass

    @class_errors
    def write_to_backtest(self, strategy_full_name, interval, result,
                          kind, fast, slow, end_result, tp_sl):
        try:
            processor.process_backtest(
                symbol=self.symbol,
                strategy_short_name=strategy_full_name[:6],
                strategy_full_name=strategy_full_name,
                interval=interval,
                result=result,
                kind=kind,
                fast=fast,
                slow=slow,
                end_result=end_result,
                tp_sl=tp_sl
            )
        except Exception as e:
            print("write_to_backtest", e)
            pass

    @class_errors
    def too_much_risk(self):
        test = [i - timedelta(minutes=5) < dt.now() < i + timedelta(minutes=45) for i in hardcore_hours]
        if any(test):
            print("High volatility risk.")
            return 1
        return 1# if not Bot.target_class.checkTarget() else 2

    @class_errors
    def volume_metrics_data(self, name_):
        def mini_data():
            with open(f'{name_}.json', "r") as file:
                data = json.load(file)
            for i in data:
                if i[0] == self.symbol:
                    return i
            return []

        my_data = mini_data()
        if my_data == []:
            return None
        my_data[1] = globals()[my_data[1]]
        print(my_data)
        return my_data

    @class_errors
    def volume_reducer(self, pos_type, name_):
        try:
            if_not_ok = 0.6
            if_ok = 1
            try:
                data = self.volume_metrics_data(name_)
            except Exception:
                return if_not_ok
            if data == []:
                return if_not_ok
            strategy = data[1]
            fast = data[2]
            slow = data[3]
            df_raw = get_data(self.symbol, data[4], 1, 5000)
            df = strategy(df_raw.copy(), slow, fast, self.symbol)[0]
            position = int(0) if df['stance'].iloc[-1] == 1 else int(1) if df['stance'].iloc[-1] == -1 else None
            if data[5] == -1 and position in [0, 1]:
                print(f"Reverse mask pos {data[4]}")
                position = int(0) if position == 1 else int(1)
            self.trend = "Long" if position == 0 else "SHORT"
            if position is None:
                print(f'volume_reducer {name_} not ok')
                return if_not_ok
            if pos_type == position:
                print(f'volume_reducer {name_} ok')
                return if_ok
            print(f'volume_reducer {name_} not ok')
            return if_not_ok
        except Exception:
            return if_not_ok

    @class_errors
    def load_strategies_from_json(self):
        with open("fast.json", "r", encoding="utf-8") as f:
            loaded_data = json.load(f)
        output = [i for i in loaded_data if i[0] == self.symbol]
        output.sort(key=lambda x: x[2])
        for i in output:
            print(i)
        intervals = np.unique([i[2] for i in output])
        intervals.sort()
        return output, intervals[::-1]

    @class_errors
    def sort_strategies(self):

        # 0- name_, 1- strategy_, 2- interval, 3- fast, 4- slow, 5- round(result, 2), 6- actual_condition,
        # 7- kind, 8- daily_return, 9- end_result, 10- tp_std, 11- sl_std, 12- drift, 13- p_value, 14- volume_contition,
        # 15- today_direction

        for i in range(len(self.strategies)):
            strategyy = self.strategies[i][1]
            intervall = self.strategies[i][2]
            fastt = self.strategies[i][3]
            sloww = self.strategies[i][4]
            self.strategies[i][8] = self.calc_pos_condition(strategyy(get_data(self.symbol, intervall, 1, 5000), sloww, fastt, self.symbol)[0])[-1]

        if dt.now().hour >= change_hour or (not self.virgin_test):# all([i[6] == -2 for i in self.strategies]) :
            self.strategies = [i for i in self.strategies if i[8] > 0 and i[5] > 0 and i[13] > 0]
            sorted_data = sorted(self.strategies, key=lambda x: x[8]*x[5]*x[13]*(x[10]/x[11]), reverse=True)
        else:
            self.strategies = [i for i in self.strategies if i[8] > 0 and i[5] > 0 and i[13] > 0]
            sorted_data = sorted(self.strategies, key=lambda x: x[5]*x[13]*(x[10]/x[11]), reverse=True)
        first_ = sorted(self.strategies, key=lambda x: x[5]*x[13]*(x[10]/x[11]), reverse=True)[0][7]
        printer("Daily starter", first_)
        self.actual_today_best = first_
        second_ = 'trend' if first_ == 'counter' else 'counter'
        group_t = [item for item in sorted_data if item[7] == first_] # first
        group_n = [item for item in sorted_data if item[7] == second_] # second
        alternating_data = []
        max_len = max(len(group_t), len(group_n))

        for i in range(max_len):
            if i < len(group_t):
                alternating_data.append(group_t[i])
            if i < len(group_n):
                alternating_data.append(group_n[i])

        self.virgin_test = False
        return alternating_data

    @measure_time
    @class_errors
    def test_strategies(self, add_number=0):
        self.close_request(True)
        strategies_number = 11 + add_number
        super_start_time = time.time()
        strategies, intervals_ = self.load_strategies_from_json()
        metric_name = strategies[0][3]
        self.bt_metric = globals()[metric_name]
        self.strategies_raw = []

        dfperms = PermutatedDataFrames(self.symbol, intervals_, int(self.number_of_bars_for_backtest))
        permutated_dataframes = dfperms.dataframes_output()

        i = 1
        for strategy in strategies:
            self.is_this_the_end()
            #self.check_trigger(backtest=True)
            name_ = strategy[1]
            strategy_ = globals()[name_]
            interval = strategy[2]
            kind = name_.split('_')[-1]
            today_direction = strategy[4+dt.now().weekday()]
            print(f'\n\nStrategy {i} from {len(strategies)}')
            i += 1
            results_pack = self.trend_backtest(strategy_, interval)
            if results_pack is None:
                print("This strategy have no results.")
                continue
            fast, slow, result, actual_condition, daily_return, end_result, risk_data = results_pack
            tp, sl, tp_std, sl_std, drift, volume_contition = risk_data
            printer("TP/TP_STD", f'{tp, tp_std}')
            printer("SL/SL_STD", f'{sl, sl_std}')
            print(name_, interval, fast, slow, round(result, 4), actual_condition, daily_return, end_result, drift, "\n")
            monte = Montecarlo(self.symbol, interval, strategy_, self.bt_metric, int(self.number_of_bars_for_backtest), slow, fast, permutated_dataframes)
            p_value = monte.final_p_value(self.avg_vol)
            printer("Z-score*1/p-value:", p_value)
            printer("Result:", result)
            printer("Daily return:", daily_return)
            printer("Final sort result virgin: ", round(p_value*result, 6))
            printer("Final sort result daily: ", round(p_value*daily_return*result, 6))
            printer("volume_contition: ", volume_contition)
            self.strategies_raw.append([name_, strategy_, interval, fast, slow, round(result, 8), actual_condition,
                                        kind, daily_return, end_result, tp_std, sl_std, drift, p_value, volume_contition, today_direction])

        print("\nv NICE STRATEGIES v")
        for strat in self.strategies_raw:
            if strat[13] > 0:
                print(strat)
        print("^ NICE STRATEGIES ^\n")

        self.backtest_time = dt.now()

        for name_, _, interval, fast, slow, result, _, kind, _, end_result, tp_std, sl_std, drift, p_value, volume_contition, today_direction in self.strategies_raw:
            try:
                tp_sl = round(tp_std/sl_std, 3)
            except ZeroDivisionError:
                tp_sl = 0
            if end_result is None:
                end_result = 0
            if p_value > 0:
                self.write_to_backtest(name_, interval, result, kind, fast, slow, end_result, tp_sl)

        self.strategies = [i for i in self.strategies_raw if ((i[5] != np.inf) and (i[5] > 0))]
        try:
            self.strategies = self.sort_strategies()
        except Exception:
            sleep(1800)
            self.test_strategies()

        time_info(time.time()-super_start_time, 'Total duration')

        # use only four best strategies
        if len(self.strategies) > strategies_number:
            self.strategies = self.strategies[:strategies_number]

        if len(self.strategies) == 0:
            self.close_request()
            print("You don't have any strategy to open position right now. Waiting a half an hour for backtest.")
            sleep(1800)
            self.test_strategies()
        else:
            for strategy in self.strategies:
                print([strategy[n] for n in range(len(strategy)) if n!=1])
            self.strategy_number = 0
            self.tiktok = 0
            self.reset_bot()

    @class_errors
    def trend_backtest(self, strategy, interval):
        print(strategy.__name__)
        #sharpe_multiplier = interval_time_sharpe(interval)
        df_raw = get_data(self.symbol, interval, 1, self.number_of_bars_for_backtest)

        results = []
        results_raw = []
        if Bot.montecarlo_for_all:
            dfperms_mini = PermutatedDataFrames(self.symbol, [interval], int(self.number_of_bars_for_backtest), how_many=100)
            permutated_dataframes_mini = dfperms_mini.dataframes_output()

        for slow in trange(5, slow_range, 2):
            for fast in range(2, fast_range, 2):
                try:
                    if fast == slow:
                        continue
                    df1, _ = strategy(df_raw, slow, fast, self.symbol)
                    if len(df1) < self.number_of_bars_for_backtest/2:
                        continue
                    df1, density = calculate_strategy_returns(df1, leverage)
                    if (len(df1) < self.number_of_bars_for_backtest/2) or (density < 1/500) or (density > 0.2):
                        continue
                    # result = self.bt_metric(df1)
                    df1['date_xy'] = df1['time'].dt.date
                    result, end_result, risk_data = calc_result_metric(df1, self.bt_metric, False, True)

                    if result > 0:
                        results_raw.sort()
                        if Bot.montecarlo_for_all:
                            if len(results_raw) > 1:
                                if len(results_raw) > 8:
                                    if result > min(results_raw[-7:]):
                                        results_raw.append(result)
                                    else:
                                        continue
                                else:
                                    if result > min(results_raw):
                                        results_raw.append(result)
                                    else:
                                        continue
                            else:
                                results_raw.append(result)
                        else:
                            if len(results_raw) > 1:
                                if result > max(results_raw):
                                    results_raw.append(result)
                                else:
                                    continue
                            else:
                                results_raw.append(result)

                        _, actual_condition, _, daily_return = self.calc_pos_condition(df1)
                        if Bot.montecarlo_for_all:
                            monte_mini = Montecarlo(self.symbol, interval, strategy, self.bt_metric, int(self.number_of_bars_for_backtest), slow, fast, permutated_dataframes_mini, how_many=100, print_tqdm=False)
                            p_value = monte_mini.final_p_value(self.avg_vol)
                            if p_value > 0:
                                #print(f"\nAdd result {fast} {slow} {result} {p_value}")
                                results.append([fast, slow, round(result*p_value, 10), actual_condition, daily_return, end_result, risk_data])
                            else:
                                continue
                        else:
                            results.append([fast, slow, round(result, 10), actual_condition, daily_return, end_result, risk_data])
                    else:
                        continue
                except Exception as e:
                    print("\ntrend_backtest", e)
                    continue

        if len(results) < 4:
            return None

        try:
            results = find_density_results(results)
            results = [i for i in results if i[2] > 0]
            f_result = sorted(results, key=lambda x: x[2], reverse=True)[0]
        except IndexError:
            return None
        print(f"Best ma factors fast={f_result[0]} slow={f_result[1]}")
        return f_result[0], f_result[1], round(f_result[2]*100, 8), f_result[3], f_result[4], f_result[5], f_result[6]


    @class_errors
    def change_tp_sl(self, tp_profit, capacity_condition=False):
        try:
            info = mt.symbol_info(self.symbol)
            positions = mt.positions_get(symbol=self.symbol)
            pos_ = [i for i in positions if i.magic == self.magic][0]
            type_to_rsi = 0 if pos_.type == 1 else 1
            new_tp = 0.0
            new_sl = 0.0

            def sltprequest(new_tp, new_sl, pos_):
                request = {
                        "action": mt.TRADE_ACTION_SLTP,
                        "symbol": self.symbol,
                        "position": pos_.ticket,
                        "magic": self.magic,
                        "tp": new_tp,
                        "sl": new_sl,
                        }
                order_result = mt.order_send(request)
                print(order_result)

            if capacity_condition:
                if pos_.sl == 0.0:
                    if pos_.type == 0:
                        new_sl = round((1-self.avg_vol/20)*info.ask, info.digits)
                    elif pos_.type == 1:
                        new_sl = round((1+self.avg_vol/20)*info.bid, info.digits)
                    sltprequest(new_tp, new_sl, pos_)
            else:
                if pos_.tp == 0.0:
                    if rsi_condition_for_tpsl(self.symbol, type_to_rsi, self.interval):
                        if pos_.type == 0:
                            if pos_.profit > tp_profit/10:
                                new_tp = round(((1+self.avg_vol/5)*info.ask), info.digits)
                                new_sl = round((pos_.price_open + info.ask*2)/3, info.digits)
                            elif pos_.profit < 0:
                                new_tp = round(((1+self.avg_vol/10)*pos_.price_open), info.digits)
                                new_sl = round((1-self.avg_vol/20)*info.ask, info.digits)
                        elif pos_.type == 1:
                            if pos_.profit > tp_profit/10:
                                new_tp = round(((1-self.avg_vol/5)*info.bid), info.digits)
                                new_sl = round((pos_.price_open + info.bid*2)/3, info.digits)
                            elif pos_.profit < 0:
                                new_tp = round(((1-self.avg_vol/10)*pos_.price_open), info.digits)
                                new_sl = round((1+self.avg_vol/20)*info.bid, info.digits)
                        sltprequest(new_tp, new_sl, pos_)

        except Exception as e:
            print("change_tp_sl", e)


    def new_strategy(self):

        del self.strategies[0]

        if len(self.strategies) == 0:
            self.test_strategies()
        elif len(self.strategies) == 1:
            pass
        else:
            self.strategies = self.sort_strategies()

if __name__ == '__main__':
    print('Yo, wtf?')
