import time
import config
import binance_futures
from termcolor import colored

def get_position_info(): # >>> "LONGING" // "SHORTING" // "NO_POSITION"
    title = "CURRENT POSITION :   "

    response = binance_futures.position_information()[0]
    positionAmt = float(response.get('positionAmt'))
    unRealizedProfit = round(float(response.get('unRealizedProfit')), config.round_decimal)
    # print(binance_futures.position_information()[0])

    if (positionAmt > 0):
        position = "LONGING"
        print(colored(title + position, "green"))
        if unRealizedProfit > 0: print(colored("unRealizedProfit :   " + str(unRealizedProfit), "green"))
        else: print(colored("unRealizedProfit :   " + str(unRealizedProfit), "red"))

    elif (positionAmt < 0):
        position = "SHORTING"
        print(colored(title + position, "red"))
        if unRealizedProfit > 0: print(colored("unRealizedProfit :   " + str(unRealizedProfit), "green"))
        else: print(colored("unRealizedProfit :   " + str(unRealizedProfit), "red"))

    else:
        position = "NO_POSITION"
        print(title + position)

    return position

def get_unRealizedProfit():
    response         = binance_futures.position_information()[0]
    markPrice        = round(float(response.get("markPrice")), config.round_decimal)
    positionAmt      = abs(round(float(response.get("positionAmt")), config.round_decimal))
    unRealizedProfit = round(float(response.get('unRealizedProfit')), config.round_decimal)
    taker_maker_fees = 0.08 # One transaction is 0.04, buy and sell means 0.04 * 2 = 0.08
    breakeven_USDT   = (markPrice * positionAmt * taker_maker_fees) / 100

    if unRealizedProfit > breakeven_USDT: return "PROFIT"
    else: return "LOSS"
