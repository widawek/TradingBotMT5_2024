import MetaTrader5 as mt
mt.initialize()

# Połącz się z terminal'em MetaTrader 5
if not mt.initialize():
    print("Nie można połączyć się z MetaTrader 5!")
    mt.shutdown()
    quit()

# Zdefiniuj symbol
symbol = "ETHUSD"

if mt.positions_get(symbol=symbol) != ():
    print("Masz otwarte pozycje na danym symbolu!")
    a = input("Czy chcesz kontunuować? Y/N ")
    if a != "Y" and a != "y":
        input("KONIEC")
        exit()


# Pobierz wszystkie zlecenia oczekujące
orders = mt.orders_get(symbol=symbol)
counter = 0
if orders is None:
    print("Brak zleceń oczekujących dla symbolu", symbol)
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


print(f"Usunięto łącznie {counter} zleceń na symbolu {symbol}")
input()
