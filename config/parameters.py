import sys
sys.path.append("..")

symbols: list = [
    'EURJPY',
    'EURUSD',
    'GBPUSD',
    'USDCHF',
    'USDJPY',
    'US30',
    'AUDUSD',
    'CADJPY',
]

# dzwignia do backtestów
leverage = 8

# godzina startu backtestów dla każdego dnia
morning_hour = 7

# godzina końca backtestów każdego dnia
evening_hour = 21

# ustala wielkosć pozycji wg wzoru (100/ilość symboli)*(position_size_float/divider) gdzie divider dla konta w usd wynosi 1 a dla złotówki kurs USDPLN
position_size_float = 0.6

# mnożnik średniej dziennej zmienności powyżej której (po osiągnięciu takiej straty) stratna pozycja będzie bezwzględnie zamknięta
kill_multiplier = 1.6

# dzielnik do funkcji wyliczania dzielnika średniej dziennej zmienności który będzie docelowym zyskiem (uniwersalnym dla każdej strategii)
trigger_model_divider_factor = 25

# stosunek zysku z otwartych pozycji do ich margin powyżej którego pozycje będą śledzone w celu zainkasowania zysku # klasa GlobalProfitTracker
global_tracker_multiplier = 0.07

# jeżeli pozycje w klasie GlobalProfitTracker będą śledzone po przekoczeniu bariery po spadku zysku poniżej tego ratio od max zostaną zamknięte żeby zainkasować zysk
profit_decrease_barrier = 0.85

# przeliczanie returns z godzin pomiędzy evening_hour a morning_hour w backtestach
respect_overnight = True

# maksymalna wartość czynnika fast w backtestach
fast_range = 21

# maksymalna wartość czynnika slow w backtestach
slow_range = 63

# backtest - mnożnik kosztu otwarcia pozycji który historycznie podaje broker
spread_multiplier = 3

# liczba swieczek dla kazdego backtestu
bot_backtest_bars = 17000

# włącza testy montecarlo na małej próbie (80-120) dla każdej kombinacji w backtest (znaczne wydłużenie czasu trwania testów)
montecarlo_for_all = False

# włącza odwrócenie pozycji zależenie od aktualnej korelacji cen wszystkich handlowancyh symboli
reverse_position_because_of_correlaton = False

# włacza bezwzględne odwrócenie pozycji (odrwaca pozycję ponad reverse_position_because_of_correlaton)
just_reverse_position = True

# dzielnik do ustawienia takeprofit dla wyliczonego tp_money dla zyskownej pozycji (np im mniejsza wartość tym wyższa wartość takeprofit dla pozycji long)
tp_divider_for_loser = 8.0

# stosunek tp/sl dla pozycji stratnej
tp_sl_loser_ratio = 2.0

# maksymalny dzielnik średniej dziennej zmienności dla ustawienia takeprofit (im większa wartość tym mniejszy takeprofit)
tp_divider_max = 5.0

# normalny dzielnik średniej dziennej zmienności dla ustawienia takeprofit (im mniejsza wartość tym wyższy takeprofit)
tp_divider_normal = 3.0

# dzielnik wyliczonego takeprofit (im mniejsza wartość tym większy wymagany profit)
tp_profit_to_pos_divider = 8.0

# okresla wagę tp do aktualnej ceny podczas zblizania takeprofit do ceny (im mniejsza wartość tym szybsze zbliżenie)
tp_weight = 6.0

# Wartość przez który zostanie przemnożony czas w funkcji generującej czas przerwy jeżeli był stoploss lub takeprofit
time_after_sl_mul = 1.5

