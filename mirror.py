import MetaTrader5 as mt
from time import sleep
mt.initialize()
commment = 'mirror'

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


def request(symbol, posType, volume):
    comment = 'mirror'
    price = mt.symbol_info(symbol).bid if posType==0 else mt.symbol_info(symbol).ask
    request = {
        "action": actions['deal'],
        "symbol": symbol,
        "volume": volume,
        "type": posType,
        "price": float(price),
        "deviation": 20,
        "magic": 777,
        "tp": 0.0,
        "sl": 0.0,
        "comment": comment,
        "type_time": mt.ORDER_TIME_GTC,
        "type_filling": mt.ORDER_FILLING_IOC,
        }
    order_result = mt.order_send(request)
    print(order_result)


def close_request_only(position):
    request = {"action": actions['deal'],
            "symbol": position.symbol,
            "volume": float(position.volume),
            "type": 1 if (position.type == 0) else 0,
            "position": position.ticket,
            "magic": position.magic,
            'deviation': 20,
            "type_time": mt.ORDER_TIME_GTC,
            "type_filling": mt.ORDER_FILLING_IOC
            }
    order_result = mt.order_send(request)
    print(order_result)


numbers_ = [str(x) for x in range(1, 9)]

def mirror():
    multiplier = 4 #int(input("Wprowadz mnoznik (int): "))
    while True:
        positions = mt.positions_get()
        for i in positions:
            if i.comment.count("_") == 3 and i.comment[-2] in numbers_:
                type_ = i.type
                volume_ = i.volume
                dig = mt.symbol_info(i.symbol).digits

                mirror_type = int(0) if type_ == 1 else int(1)
                mirror_volume = round(multiplier*volume_, dig)

                if any([((n.symbol == i.symbol) and (n.comment == commment) and (n.type == mirror_type)) for n in positions]): # mirror position is open
                    pass
                elif any([((n.symbol == i.symbol) and (n.comment == commment) and (n.type == type_)) for n in positions]): # mirror position is open in the same direction
                    close_request_only([n for n in positions if ((n.symbol == i.symbol) and (n.comment == commment) and (n.type == type_))][0])
                else:
                    request(i.symbol, mirror_type, mirror_volume)

            if i.comment == commment:
                if any([(n.comment.count("_") == 3 and n.comment[-2] in numbers_ and i.symbol == n.symbol) for n in positions]):
                    pass
                else:
                    position = [n for n in positions if (n.comment == commment and n.symbol == i.symbol)][0]
                    print(position)
                    close_request_only(position)
        sleep(0.5)
