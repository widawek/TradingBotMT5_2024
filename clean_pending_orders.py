import MetaTrader5 as mt
mt.initialize()


def close_request(symbol):
    if symbol == "ALL":
        positions_ = mt.positions_get()
    else:
        positions_ = mt.positions_get(symbol=symbol)
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
        print(order_result)


def close_pendings(symbol, positions=True):
    if positions:
        close_request(symbol)

    if symbol == "ALL":
        orders = mt.orders_get()
    else:
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


if __name__ =='__main__':
    symbol = "ALL"
    close_pendings(symbol, positions=True)
    input()
