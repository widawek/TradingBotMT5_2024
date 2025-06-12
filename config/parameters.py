import sys
sys.path.append("..")


symbols: list                                   = [
                                                    'EURJPY',
                                                    'EURUSD',
                                                    'GBPUSD',
                                                    'USDCHF',
                                                    'USDJPY',
                                                    'US30',
                                                    'AUDUSD',
                                                    'CADJPY',
                                                    "AMD",
                                                    "AAPL"
                                                ]

# majors
leverage: int                                   = 8         # dzwignia do backtestów
morning_hour: int                               = 7         # godzina startu backtestów dla każdego dnia
evening_hour: int                               = 21        # godzina końca backtestów każdego dnia
position_size_float: float                      = 0.6       # ustala wielkosć pozycji wg wzoru (100/ilość symboli)*(position_size_float/divider) gdzie divider dla konta w usd wynosi 1 a dla złotówki kurs USDPLN
kill_multiplier: float                          = 1.6       # mnożnik średniej dziennej zmienności powyżej której (po osiągnięciu takiej straty) stratna pozycja będzie bezwzględnie zamknięta
trigger_model_divider_factor: int               = 11        # dzielnik do funkcji wyliczania dzielnika średniej dziennej zmienności który będzie docelowym zyskiem (uniwersalnym dla każdej strategii)
global_tracker_multiplier: float                = 0.05      # stosunek zysku z otwartych pozycji do ich margin powyżej którego pozycje będą śledzone w celu zainkasowania zysku # klasa GlobalProfitTracker
profit_decrease_barrier: float                  = 0.85      # jeżeli pozycje w klasie GlobalProfitTracker będą śledzone po przekoczeniu bariery po spadku zysku poniżej tego ratio od max zostaną zamknięte żeby zainkasować zysk
respect_overnight: bool                         = True      # przeliczanie returns z godzin pomiędzy evening_hour a morning_hour w backtestach
fast_range: int                                 = 21        # maksymalna wartość czynnika fast w backtestach
slow_range: int                                 = 63        # maksymalna wartość czynnika slow w backtestach
spread_multiplier: int                          = 4         # backtest - mnożnik kosztu otwarcia pozycji który historycznie podaje broker
bot_backtest_bars: int                          = 17000     # liczba swieczek dla kazdego backtestu
montecarlo_for_all: bool                        = False     # włącza testy montecarlo na małej próbie (80-120) dla każdej kombinacji w backtest (znaczne wydłużenie czasu trwania testów)
reverse_position_because_of_correlaton: bool    = True      # włącza odwrócenie pozycji zależenie od aktualnej korelacji cen wszystkich handlowancyh symboli
just_reverse_position: bool                     = True      # włacza bezwzględne odwrócenie pozycji (odrwaca pozycję ponad reverse_position_because_of_correlaton)

# minors
tp_divider_for_loser: float                     = 8.0       # dzielnik do ustawienia takeprofit dla wyliczonego tp_money dla zyskownej pozycji (np im mniejsza wartość tym wyższa wartość takeprofit dla pozycji long)
tp_sl_loser_ratio: float                        = 2.0       # stosunek tp/sl dla pozycji stratnej
tp_divider_max: float                           = 5.0       # maksymalny dzielnik średniej dziennej zmienności dla ustawienia takeprofit (im większa wartość tym mniejszy takeprofit)
tp_divider_normal: float                        = 3.0       # normalny dzielnik średniej dziennej zmienności dla ustawienia takeprofit (im mniejsza wartość tym wyższy takeprofit)
tp_profit_to_pos_divider: float                 = 8.0       # dzielnik wyliczonego takeprofit (im mniejsza wartość tym większy wymagany profit)
tp_weight: float                                = 6.0       # okresla wagę tp do aktualnej ceny podczas zblizania takeprofit do ceny (im mniejsza wartość tym szybsze zbliżenie)
