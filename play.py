from simple_random_bot import Control, Bot
import MetaTrader5 as mt
mt.initialize()

# symbols = ["ETHUSD", "JP225", "XAGAUD", "XAGEUR", "UK100"]
symbol = "JP225"
# a = Control(symbols)
# a.control_panel()
limit_price = 38420
bot = Bot(symbol, now=False, limit_price=limit_price)
bot.report()
