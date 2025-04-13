import pandas as pd
import MetaTrader5 as mt
import sys
from datetime import datetime as dt
from app.functions import *
from config.parameters import *
from strategies_analize.global_strategies import *
from strategies_analize.metrics import *
from app.bot_functions import *
sys.path.append("..")
mt.initialize()


class Reverse:
    def __init__(self, symbol):
        self.symbol = symbol
        self.condition = False# if dt.now().weekday() != 0 else True
        self.one_percent_balance = -10 * round(mt.account_info().balance/100, 2)

    def closed_pos(self, symbol: str = 'all'):
        dzisiaj = dt.now().date()
        poczatek_dnia = dt.combine(dzisiaj, dt.min.time())
        koniec_dnia = dt.combine(dzisiaj, dt.max.time())
        zamkniete_transakcje = mt.history_deals_get(poczatek_dnia, koniec_dnia)
        if zamkniete_transakcje is None or len(zamkniete_transakcje) == 0:
            return 0
        # Filtrujemy transakcje po podanym symbolu
        if symbol != 'all':
            zamkniete_transakcje = [deal for deal in zamkniete_transakcje if deal.symbol == symbol]
        if not zamkniete_transakcje:
            return 0
        profit_list = [deal.profit for deal in zamkniete_transakcje if deal.profit != 0]
        suma_zyskow = sum(profit_list)
        printer(f"\nSuma zysków z zamkniętych pozycji dla {symbol} dzisiaj:",  f"{suma_zyskow:.2f} PLN")
        return suma_zyskow, profit_list

    def reverse_or_not(self):
        try:
            if self.condition:
                return self.condition
            factor = 3
            std_ = 3
            cp = self.closed_pos()
            df = pd.DataFrame({'profits': cp[1]})
            if len(df) < 5:
                return self.condition
            df['profits_sum'] = df['profits'].cumsum()
            df['mean_profits'] = df['profits_sum'].expanding().mean()
            df['profits_sum_mean'] = df['profits_sum'].rolling(factor).mean()
            df['profits_sum_std'] = std_*df['profits_sum'].rolling(factor).std()
            df['boll_up'] = df['profits_sum_mean'] + df['profits_sum_std']
            df['boll_down'] = df['profits_sum_mean'] - df['profits_sum_std']
            df['cond'] = (df['boll_up'] < df['mean_profits'])&(df['boll_up'].shift() < df['mean_profits'].shift())
            try:
                pos = [i for i in mt.positions_get() if i.symbol == self.symbol]
                profit_symbol = pos[0].profit
                printer("Aktualny zysk z otwartych pozycji:", f"{profit_symbol:.2f} PLN")
            except Exception:
                profit_symbol = 0

            symbol_profit, symbol_profits = self.closed_pos(self.symbol)
            if len(symbol_profits) < 2:
                return self.condition

            if df['cond'].iloc[-1] and symbol_profit < 0 and profit_symbol <= 0 and symbol_profits[-1] < 0 and cp[0] < self.one_percent_balance:
                self.condition = True
            return self.condition
        except Exception:
            return self.condition


class Target:
    def __init__(self, target=0.05):
        self.start_balance = mt.account_info().balance - closed_pos()
        self.target = target
        self.result = False
        self.last_self_result = False
        self.test = 0

    def checkTarget(self):
        self.last_self_result = self.result
        if not self.result:
            actual_result = mt.account_info().balance + sum([i.profit for i  in mt.positions_get()])
            if actual_result > self.start_balance * (1+self.target):
                self.result = True
        if self.last_self_result != self.result:
            self.test += 1
        if self.test > 1:
            return self.result, False
        return self.result, self.last_self_result != self.result


class GlobalProfitTracker:
    mt.initialize()
    def __init__(self, symbols: list, multiplier: float):
        self.barrier = round((len(symbols)*multiplier)/(100), 5)
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