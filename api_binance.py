import os, time, config, requests
from webhook_launcher import telegram_bot_sendtext
from binance.client import Client
from termcolor import colored

# Get environment variables
api_key     = ""
api_secret  = ""
client      = Client(api_key, api_secret)
live_trade  = config.live_trade

#To send webhook or telegram notification
active_webhook = True

def get_timestamp():
    return int(time.time() * 1000)

def position_information(pair):
    time.sleep(1)
    try:
        return client.futures_position_information(symbol=pair, timestamp=get_timestamp(), recvWindow=5000)
    except Exception as e:
        print(f"Error getting position information: {e}")
        return None

def account_trades(pair, timestamp):
    time.sleep(1)
    return client.futures_account_trades(symbol=pair, timestamp=get_timestamp(), startTime=timestamp, recvWindow=5000)

def LONG_SIDE(response):
    time.sleep(1)
    if float(response[1].get('positionAmt')) > 0: return "LONGING"
    elif float(response[1].get('positionAmt')) == 0: return "NO_POSITION"

def SHORT_SIDE(response):
    time.sleep(1)
    if float(response[2].get('positionAmt')) < 0 : return "SHORTING"
    elif float(response[2].get('positionAmt')) == 0: return "NO_POSITION"

def change_leverage(pair, leverage):
    return client.futures_change_leverage(symbol=pair, leverage=leverage, timestamp=get_timestamp(), recvWindow=5000)

def change_margin_to_ISOLATED(pair):
    return client.futures_change_margin_type(symbol=pair, marginType="ISOLATED", timestamp=get_timestamp(), recvWindow=5000)

def set_hedge_mode():
    if not client.futures_get_position_mode(timestamp=get_timestamp()).get('dualSidePosition'):
        return client.futures_change_position_mode(dualSidePosition="true", timestamp=get_timestamp(), recvWindow=5000)

def market_open_long(pair, quantity):
    time.sleep(1)
    if live_trade:
        client.futures_create_order(symbol=pair,
                                    quantity=config.quantity_long_sep,
                                    positionSide="LONG",
                                    type="MARKET",
                                    side="BUY",
                                    timestamp=get_timestamp())
    print(colored("GO_LONG", "green"))
    if active_webhook:
        telegram_bot_sendtext(" GO_LONG "+ str(pair) + " "+ str(config.quantity_long_sep) + " BUY MARKET ")





def market_open_short(pair, quantity):
    time.sleep(1)
    if live_trade:
        client.futures_create_order(symbol=pair,
                                    quantity=config.quantity_short_sep,
                                    positionSide="SHORT",
                                    type="MARKET",
                                    side="SELL",
                                    timestamp=get_timestamp())
    print(colored("GO_SHORT", "red"))
    if active_webhook:
        telegram_bot_sendtext(" GO_SHORT "+ str(pair) + " "+ str(config.quantity_short_sep) + " SELL MARKET ")


def market_close_long(pair, response):
    if live_trade:
        client.futures_create_order(symbol=pair,
                                    quantity=abs(float(response[1].get('positionAmt'))),
                                    positionSide="LONG",
                                    side="SELL",
                                    type="MARKET",
                                    timestamp=get_timestamp())
    print("CLOSE_LONG")
    if active_webhook:
        telegram_bot_sendtext(" CLOSE_LONG "+str(pair)+ " | Position: "+ str(abs(float(response[1].get('positionAmt')))) + "| X"+ str(response[1].get('leverage')) + " | Market Price: "+ str(float(response[1].get('markPrice'))) + " Profit: "+ str(float(response[1].get('unRealizedProfit'))) + " SELL MARKET ")

def market_close_short(pair, response):
    if live_trade:
        client.futures_create_order(symbol=pair,
                                    quantity=abs(float(response[2].get('positionAmt'))),
                                    positionSide="SHORT",
                                    side="BUY",
                                    type="MARKET",
                                    timestamp=get_timestamp())
    print("CLOSE_SHORT")
    if active_webhook:
        telegram_bot_sendtext(" CLOSE_SHORT "+pair+" | Position: "+ str(abs(float(response[2].get('positionAmt')))) + "| X"+ str(response[2].get('leverage')) + " | Market Price: "+ str(float(response[2].get('markPrice'))) + " Profit: "+ str(float(response[2].get('unRealizedProfit'))) + "   BUY MARKET ")

set_hedge_mode()
