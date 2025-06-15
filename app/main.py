import pandas as pd
import numpy as np
import MetaTrader5 as mt
import time
import string
import sys
import os
from extensions.investing_scrapper import Scraper
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
alphabet = list(string.ascii_lowercase)


def divider():
    import MetaTrader5 as mt
    mt.initialize()
    curr = mt.account_info().currency
    df_ = get_data("USDPLN", "D1", 0, 1)
    last_o = float(df_['open'].iloc[-1])
    divider = 1 if curr == 'USD' else last_o
    return divider


if dt.now().weekday() not in [5, 6]:
    scraper = Scraper()
    hardcore_hours: list = [] if dt.now().weekday() in [5, 6] else scraper.give_me_hardcore_hours()


position_size: float = round((100/len(symbols))*(position_size_float/divider()), 3)
tz_diff: int = get_timezone_difference()


class Bot:
    print('montecarlo_for_all', montecarlo_for_all)
    target_class = Target()
    weekday = dt.now().weekday()
    def __init__(self, symbol):
        self.actual_mirror = 0
        self.montecarlo_for_all = montecarlo_for_all
        self.checkout_stoploss = True
        self.position_capacity = []
        self.backtest_time = dt.now()
        self.total_reverse = False
        self.posid = False
        #self.reverse = Reverse(symbol)
        self.if_position_with_trend = 'y'
        self.fresh_daily_target = False
        self.currency = mt.account_info().currency
        self.pwt_short, self.pwt_long, self.pwt_dev, self.pwt_divider = play_with_trend_bt(symbol)
        self.actual_today_best = 'x'
        self.use_tracker = True if symbol == symbols[0] else False
        self.positionTracker = GlobalProfitTracker(symbols, global_tracker_multiplier) if self.use_tracker else None
        self.number_of_bars_for_backtest = bot_backtest_bars
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
        self.mdv = self.mdv_()
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
    def position_time_minutes(self):
        try:
            dt_from_timestamp = dt.fromtimestamp(mt.positions_get(symbol=self.symbol)[0][1])
        except Exception:
            return 0
        tick = mt.symbol_info_tick(self.symbol)
        broker_time = dt.fromtimestamp(tick.time)
        return int((broker_time-dt_from_timestamp).seconds/60)

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
    def tiktok_slcheck(self, check_it):
        if not check_it:
            return False, 0
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
                profit = zamkniete_transakcje[-1].profit + zamkniete_transakcje[-1].commission*2
                print(comment, profit)
                self.checkout_stoploss = False
                return ('sl' in comment) or ('tp' in comment), profit
            return False, 0
        except Exception as e:
            print("tiktok_slcheck", e)
            return False, 0

    @class_errors
    def if_tiktok(self, backtest=False):
        try:
            last_pos_by_sl, last_profit = self.tiktok_slcheck(backtest)

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
            if not backtest:
                if self.tiktok < 1:
                    if ((profit_ > 0) and (last_two >= 0)):
                        self.tiktok -= 1
                    elif (profit_ < 0):
                        self.tiktok += 1
                    else:
                        pass
                else:
                    if ((profit_ > 0) and (last_two >= 0)):
                        self.tiktok -= 1
                    else:
                        self.new_strategy()
                        self.tiktok = 0
                        self.position_size = position_size
            else:
                cond, time_ = self.last_pos_sltp()
                if cond and self.request_sltp:
                    self.request_sltp = False
                    print(f"Sleep {time_} minutes after sl or tp")
                    time.sleep(time_*60)

                if self.tiktok < 1:
                    if (last_pos_by_sl and last_profit > 0):
                        self.tiktok -= 1
                    elif (last_pos_by_sl and last_profit < 0):
                        self.tiktok += 1
                    else:
                        pass
                else:
                    if (last_pos_by_sl and last_profit > 0):
                        self.tiktok -= 1
                    else:
                        self.new_strategy()
                        self.tiktok = 0
                        self.position_size = position_size
            self.tiktok = 0 if self.tiktok < 0 else self.tiktok
            printer("Tiktok", self.tiktok)
        except Exception as e:
            print('if_tiktok', e)

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

                kind_ = self.strategies[self.strategy_number][7]
                sl = self.sl_money
                tp = round(sl*1.2, 2)
                self.self_decline_factor(tp)

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
                    condition = self.check_capacity()
                    if condition:
                        self.change_tp_sl(tp, condition)
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
    def self_decline_factor(self, tp, multiplier: float=1.22):
        min_val = 0.65
        max_val = 0.9
        min_value = 0
        max_value = tp*multiplier
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
            try:
                self.results_for_rsi_condition = rsi_condition_backtest(self.symbol, self.intervals_, leverage, 5000)
            except Exception:
                pass
            if self.checkout_stoploss:
                self.if_tiktok(self.checkout_stoploss)
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
            daily = df.iloc[-10:].copy()
            df = df[(df['hour'] > 9)&(df['hour'] < 20)]
            volatility_5 = ((df['high']-df['low'])/df['open']).mean()
            volatility_d = ((daily['high']-daily['low'])/daily['open']).mean()
            return round((volatility_5/volatility_d), 4)

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
        # volume_m15 = self.volume_reducer(posType, 'M15')
        # volume_m20 = self.volume_reducer(posType, 'M20')
        # if volume_m15 == 1 and volume_m20 == 1:
        #     self.if_position_with_trend = 'y'
        # elif volume_m15 != 1 and volume_m20 != 1:
        #     self.if_position_with_trend = 'n'
        # elif volume_m15 == 1 and volume_m20 != 1:
        #     self.if_position_with_trend = 's'
        # elif volume_m15 != 1 and volume_m20 == 1:
        #     self.if_position_with_trend = 'l'

        max_pos_margin2 = max_pos_margin * atr() * another_new_volume_multiplier_from_win_rate_condition
        max_pos_margin2 = (max_pos_margin2 + max_pos_margin2*trend_bonus)#*volume_m15*volume_m20
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

        # try:
        #     another_volume_condition = rsi_condition(self.symbol, posType, 'D1', self.results_for_rsi_condition)
        #     volume = volume*2 if another_volume_condition else volume
        #     if another_volume_condition:
        #         print("Position is ok. Go with the flow.")
        # except Exception as e:
        #     print("volume_condition - volume*2: ", e)

        if volume > symbol_info["volume_max"]:
            volume = float(symbol_info["volume_max"])
        self.volume = volume
        if min_volume or (volume < symbol_info["volume_min"]):
            self.volume = symbol_info["volume_min"]
        _, self.kill_position_profit, _ = symbol_stats(self.symbol, self.volume, kill_multiplier)
        self.kill_position_profit = round(self.kill_position_profit, 2)# * (1+self.multi_voltage('M5', 33)), 2)
        self.profit_needed = round(self.kill_position_profit/self.trigger_model_divider, 2)
        self.profit_needed_min = round(self.profit_needed / (self.volume/symbol_info["volume_min"]), 2)
        self.fresh_daily_target = False
        printer('Min volume:', min_volume)
        printer('Calculated volume:', volume)
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

            kind = strategy[15]
            position = int(0) if stance == 1 else int(1)
            mode__ = "NORMAL"

            dfx['cross'] = np.where(dfx['stance'] != dfx['stance'].shift(), 1, 0)
            self.fresh_signal = True if dfx['stance'].iloc[-1] != dfx['stance'].iloc[-2] else False
            cross = dfx[dfx['cross'] == 1]
            self.strategy_pos_open_price = cross['close'].iloc[-1]
            printer("Last open position time by MetaTrader", f"{cross['time'].iloc[-1]}", base_just=60)

            positions_ = mt.positions_get(symbol=self.symbol)
            positions_ = [i for i in positions_ if i.magic == self.magic]

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
                        case 0: self.good_price_to_open_pos = True if \
                            rsi_condition(self.symbol, 0, self.interval, self.results_for_rsi_condition, kind) else False #(price <= self.strategy_pos_open_price) and rsi_condition(self.symbol, 0) else False
                        case 1: self.good_price_to_open_pos = True if \
                            rsi_condition(self.symbol, 1, self.interval, self.results_for_rsi_condition, kind) else False #(price >= self.strategy_pos_open_price) and rsi_condition(self.symbol, 0) else False

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

        new_id = self.if_last_pos_is_bad_and_end_by_sl()
        if new_id and self.posid != new_id:
            self.posid = new_id
            self.total_reverse = True if self.total_reverse is False else False
            self.if_position_with_trend = 'y' if self.if_position_with_trend == 'n' else 'n'

        if just_reverse_position:
            if self.total_reverse:
                position = 0 if position == 1 else 1

        elif reverse_position_because_of_correlaton:
            position = self.position_reverse(position)

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
            sleep_time_after_al = int(intervals[intervals.index(self.interval)+4][1:]*time_after_sl_mul)
            comment = zamkniete_transakcje[-1].comment
            condition = ('tp' in comment) or ('sl' in comment)
            return condition, sleep_time_after_al
        except Exception as e:
            print("last_pos_sltp", e)
            return False, 30

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

        self.volume_calc(self.position_size, posType, False)

        strategy = self.strategies[self.strategy_number]
        name_ = strategy[0][:5]
        fast = strategy[3]
        slow = strategy[4]
        reverseornot = 'a'

        max_mul = 4

        try:
            profit, efficiency = get_today_closed_profit_for_symbol(self.symbol)
            if profit > 0 and efficiency > 50 and self.actual_mirror <= 0:
                self.actual_mirror = 1
            elif profit < 0 and efficiency < 50 and self.actual_mirror >= 0:
                self.actual_mirror = -1
            elif (profit < 0 and efficiency > 50) or (profit > 0 and efficiency < 50):
                self.actual_mirror = 0
            elif profit > 0 and efficiency > 50 and self.actual_mirror > 0:
                self.actual_mirror += 1
            elif profit < 0 and efficiency < 50 and self.actual_mirror < 0:
                self.actual_mirror -= 1


            # profit, efficiency = get_today_closed_profit_for_symbol(self.symbol)
            # if profit > 0 and efficiency > 50 and self.actual_mirror >= 0:
            #     self.actual_mirror = -1
            # elif profit < 0 and efficiency < 50 and self.actual_mirror >= 0:
            #     self.actual_mirror = 1
            # elif (profit < 0 and efficiency > 50) or (profit > 0 and efficiency < 50):
            #     self.actual_mirror = 0
            # elif profit > 0 and efficiency > 50 and self.actual_mirror > 0:
            #     self.actual_mirror -= 1
            # elif profit < 0 and efficiency < 50 and self.actual_mirror < 0:
            #     self.actual_mirror += 1

            if self.actual_mirror > max_mul:
                self.actual_mirror = max_mul

            if self.actual_mirror < -max_mul:
                self.actual_mirror = -max_mul

            reverseornot = alphabet[self.actual_mirror]
        except Exception as e:
            print(e)

        metric_numb = str(metric_numb_dict[self.bt_metric.__name__])
        self.comment = f'{name_}{self.interval[-1:][0]}_{fast}_{slow}_{reverseornot}{metric_numb}{self.if_position_with_trend}'


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
        self.checkout_stoploss = True
        self.request_sltp = True
        if order_result.comment == 'No money':
            seconds = 300
            print(f"No enough money to open position. Waiting {round(seconds/60, 2)} minutes")
            sleep(300)
            self.request_get()
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
            efficiency_sum = sum(self.position_capacity)/self.mdv
            print("Position capacity:  ", round(capacity, 5))
            print("Position efficiency:", round(efficiency, 3))
            print("Position efficiency_sum:", round(efficiency_sum, 5))
            if capacity < 0 and efficiency < 0.1 and efficiency_sum < -5 and self.duration():
                return 'super loss'
            if capacity < 0 and efficiency < 0.33 and efficiency_sum < -2 and self.duration(1):
                return 'loss'
            elif capacity > 0 and efficiency > 0.66 and efficiency_sum > 2 and self.duration():
                return 'profit'
            return False
        except Exception:
            return False

    @class_errors
    def duration(self, adder):
        intervals = ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M10', 'M12', 'M15', 'M20', 'M30', 'H1', 'H2']
        duration_time = int(intervals[intervals.index(self.interval)+4+adder][1:])
        duration = self.position_time_minutes()
        print(f"Duration: {duration}")
        return duration > duration_time

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
                profit_factor=1,
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
            if_not_ok = 1
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

        self.strategies = [i for i in self.strategies if i[5] > 0 and i[8] > 0 and i[13] > 0]
        sorted_data = sorted(self.strategies, key=lambda x: x[8]*x[5]*x[13]*(x[10]/x[11]), reverse=True)
        first_ = sorted(self.strategies, key=lambda x: x[8]*x[5]*x[13]*(x[10]/x[11]), reverse=True)[0][7]
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
        strategies, self.intervals_ = self.load_strategies_from_json()
        self.results_for_rsi_condition = rsi_condition_backtest(self.symbol, self.intervals_, leverage, 5000)

        metric_name = strategies[0][3]
        self.bt_metric = globals()[metric_name]
        self.strategies_raw = []

        dfperms = PermutatedDataFrames(self.symbol, self.intervals_, int(self.number_of_bars_for_backtest))
        permutated_dataframes = dfperms.dataframes_output()

        i = 1
        for strategy in strategies:
            self.is_this_the_end()
            #self.check_trigger(backtest=True)
            name_ = strategy[1]
            strategy_ = globals()[name_]
            interval = strategy[2]
            kind = name_.split('_')[-1]
            try:
                today_direction = strategy[4+dt.now().weekday()]
            except Exception as e:
                print(e)
                today_direction = 1
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
            self.montecarlo_for_all = True if self.montecarlo_for_all==False else False
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
        if self.montecarlo_for_all:
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
                        if self.montecarlo_for_all:
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
                        if self.montecarlo_for_all:
                            monte_mini = Montecarlo(self.symbol, interval, strategy, self.bt_metric, int(self.number_of_bars_for_backtest),
                                                    slow, fast, permutated_dataframes_mini, how_many=100, print_tqdm=False)
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

        if len(results) < 2:
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

        strategy = self.strategies[self.strategy_number]

        try:
            info = mt.symbol_info(self.symbol)
            digits_ = info.digits
            positions = mt.positions_get(symbol=self.symbol)
            pos_ = [i for i in positions if i.magic == self.magic][0]
            type_to_rsi = 1 if pos_.type == 1 else 0
            new_tp = 0.0
            new_sl = 0.0

            def sltprequest(new_tp, new_sl, pos_):

                if new_tp == 0.0:
                    new_tp = pos_.tp

                if new_sl == 0.0:
                    new_sl = pos_.sl

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


            if rsi_condition_for_tpsl(self.symbol, type_to_rsi, self.interval):
                if pos_.tp == 0.0:
                    if pos_.type == 0:
                        if pos_.profit > tp_profit/tp_profit_to_pos_divider:
                            if pos_.sl != 0.0 and pos_.sl > pos_.price_open:
                                tp_divider = tp_divider_normal
                            else:
                                tp_divider = tp_divider_max

                            new_tp = round(((1+self.avg_vol/tp_divider)*info.ask), digits_)
                            new_sl = round((pos_.price_open*2 + info.ask)/3, digits_)
                        else:
                            new_tp = round(((1+self.avg_vol/tp_divider_for_loser)*info.ask), digits_)
                            new_sl = round((1-self.avg_vol/(tp_divider_for_loser*tp_sl_loser_ratio))*info.ask, digits_)

                    elif pos_.type == 1:
                        if pos_.profit > tp_profit/tp_profit_to_pos_divider:
                            if pos_.sl != 0.0 and pos_.sl < pos_.price_open:
                                tp_divider = tp_divider_normal
                            else:
                                tp_divider = tp_divider_max

                            new_tp = round(((1-self.avg_vol/tp_divider)*info.bid), digits_)
                            new_sl = round((pos_.price_open*2 + info.bid)/3, digits_)
                        else:
                            new_tp = round(((1-self.avg_vol/tp_divider_for_loser)*info.bid), digits_)
                            new_sl = round((1+self.avg_vol/(tp_divider_for_loser*tp_sl_loser_ratio))*info.bid, digits_)

            elif capacity_condition and pos_.sl == 0.0:
                if pos_.sl == 0.0 or (capacity_condition == 'super loss'):

                    if capacity_condition == 'super loss':
                        self.clean_orders()

                    if pos_.type == 0:
                        if capacity_condition == 'loss':
                            #new_sl = round((1-self.avg_vol/10)*info.ask, digits_) # strategy[11]
                            new_sl = round((1-strategy[11]/tp_divider_max)*info.ask, digits_)
                        elif capacity_condition == 'profit':
                            #new_sl = round((1-self.avg_vol/20)*pos_.price_open, digits_)
                            new_sl = round((1-strategy[11]/tp_divider_normal)*info.ask, digits_)

                    elif pos_.type == 1:
                        if capacity_condition == 'loss':
                            #new_sl = round((1+self.avg_vol/10)*info.bid, digits_)
                            new_sl = round((1+strategy[11]/tp_divider_max)*info.bid, digits_)
                        elif capacity_condition == 'profit':
                            #new_sl = round((1+self.avg_vol/20)*pos_.price_open, digits_)
                            new_sl = round((1+strategy[11]/tp_divider_normal)*info.bid, digits_)

            elif pos_.tp != 0.0:
                try:
                    if self.tp_time < dt.now() - timedelta(minutes=round(int(self.interval[1:]))):
                        if pos_.type == 0:
                            new_tp = round((pos_.tp*tp_weight + info.ask)/(tp_weight+1), digits_)
                            if new_tp > pos_.tp:
                                new_tp = pos_.tp
                        elif pos_.type == 1:
                            new_tp = round((pos_.tp*tp_weight + info.bid)/(tp_weight+1), digits_)
                            if new_tp < pos_.tp:
                                new_tp = pos_.tp
                except Exception as e:
                    self.tp_time = dt.now()
                    print("CHANGE TP", e)

            else:
                pass

            if pos_.sl != 0.0:
                try:
                    if pos_.type == 0:
                        if pos_.sl < pos_.price_open and info.ask > pos_.price_open:
                            if pos_.profit > tp_profit/tp_profit_to_pos_divider:
                                new_sl = round((pos_.price_open*tp_weight + info.ask)/(tp_weight+1), digits_)
                                if pos_.tp != 0.0:
                                    new_tp = round(pos_.tp + abs(pos_.tp-info.ask)/2, digits_)

                        elif pos_.sl > pos_.price_open:
                            new_slx = round((pos_.price_open*tp_weight + info.ask)/(tp_weight+1), digits_)
                            if new_slx > pos_.sl:
                                new_sl == new_slx

                    elif pos_.type == 1:
                        if pos_.sl > pos_.price_open and info.ask < pos_.price_open:
                            if pos_.profit > tp_profit/tp_profit_to_pos_divider:
                                new_sl = round((pos_.price_open*tp_weight + info.bid)/(tp_weight+1), digits_)
                                if pos_.tp != 0.0:
                                    new_tp = round(pos_.tp - abs(pos_.tp-info.bid)/2, digits_)

                        elif pos_.sl < pos_.price_open:
                            new_slx = round((pos_.price_open*tp_weight + info.bid)/(tp_weight+1), digits_)
                            if new_slx < pos_.sl:
                                new_sl == new_slx

                except Exception as e:
                    print("CHANGE SL", e)

            if new_sl != 0.0 or new_tp != 0.0:
                if pos_.tp != new_tp:
                    self.tp_time = dt.now()
                sltprequest(new_tp, new_sl, pos_)

        except Exception as e:
            print("change_tp_sl", e)

    @class_errors
    def new_strategy(self):

        del self.strategies[0]

        if len(self.strategies) == 0:
            self.test_strategies()
        elif len(self.strategies) == 1:
            pass
        else:
            self.strategies = self.sort_strategies()

    @class_errors
    def position_reverse(self, position):
        try:
            df = pd.read_excel(f'{parent_catalog}\\correlation.xlsx')
        except Exception as e:
            print(e)
            try:
                df = pd.read_excel(f'{parent_catalog}\\correlation2.xlsx')
            except Exception as e:
                print(e)
                pass
            pass
        try:
            if (df['correlation_expanding'].iloc[-1] < df['correlation'].iloc[-1]) and df['correlation'].iloc[-1] > 0.01:
                return 0 if position == 1 else 1
            else:
                return position
        except Exception as e:
            print(e)
            return position


    @class_errors
    def if_last_pos_is_bad_and_end_by_sl(self):

        def groupby_profie_with_comment(symbol):
            dzisiaj = dt.now().date()
            poczatek_dnia = dt.combine(dzisiaj, dt.min.time())
            koniec_dnia = dt.now() + timedelta(days=2)
            zamkniete_transakcje = mt.history_deals_get(poczatek_dnia, koniec_dnia, group=symbol)
            if zamkniete_transakcje == ():
                return None
            zamkniete_transakcje = [i._asdict() for i in zamkniete_transakcje]
            df = pd.DataFrame(zamkniete_transakcje)
            df = df[['symbol', 'time', 'position_id', 'price', 'commission', 'profit', 'comment']]
            df = df[df.groupby('position_id')['position_id'].transform('count') > 1]
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df['comment'] = df['comment'] + " "
            x=df.groupby('position_id').agg({'commission':'sum', 'profit':'sum', 'comment':'sum'})
            x['profit'] = x['commission'] + x['profit']
            x = x[~x['comment'].str.contains("mirror", na=False)]
            x = x[['profit', 'comment']]
            return x.reset_index()

        df = groupby_profie_with_comment(self.symbol)
        if df is None:
            return False
        if 'sl ' in df['comment'].iloc[-1]:
            if df['profit'].iloc[-1] < 0:
                return df['position_id'].iloc[-1]
        return False

if __name__ == '__main__':
    print('Yo, wtf?')
