import os
import time
import random
import socket
import requests
import utils
import urllib3
import logging
from datetime import datetime
from binance.exceptions import BinanceAPIException
from termcolor import colored
from pathlib import Path
from collections import deque
import config
import strategies.combined as combined
import api_binance as api

# Logger setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Changed from ERROR to INFO for better visibility

error_dir = Path("ERROR")
error_dir.mkdir(exist_ok=True)
handler = logging.FileHandler(error_dir / "error.log", "a", encoding="utf-8")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

print(colored("------ LIVE TRADE IS ENABLED ------\n", "green")) if config.live_trade else print(colored("THIS IS BACKTESTING\n", "red"))
print(f"Current time: {datetime.now().strftime('%H:%M:%S')}")

def make_money(pair, leverage, quantity):
    response = api.position_information(pair)
    hero = combined.futures_hero(pair)
    if api.long_side(response) == "NO_POSITION" and hero["GO_LONG"].iloc[-1]:
        api.market_open_long(pair, quantity)
    else:
        logger.info("_LONG_SIDE : WAIT ")

def take_long_action(pair, response, quantity, hero, multiplier_long):
    un_realized_profit = utils.get_unrealized_profit(response)
    if utils.in_profit(response):
        api.market_close_long(pair, quantity)
    else:
        if un_realized_profit < 0 and abs(utils.calculate_profit_margin(response)) > 5:  # Threshold
            additional_quantity = utils.calculate_new_quantity(quantity, multiplier_long)
            api.market_open_long(pair, additional_quantity)
        elif utils.stop_loss_triggered(response):
            api.market_close_long(pair, quantity)
            logger.info(f"Stop Loss triggered for {pair} long position. Closed position.")

def process_long_position(pair, response, quantity, hero, leverage, i):
    multiplier_long = config.multiplier_long[i]
    logger.info(f"Processing position for {pair}")
    response = api.position_information(pair)
    hero = combined.futures_hero(pair)
    take_long_action(pair, response, quantity, hero, multiplier_long)

def take_short_action(pair, response, quantity, hero, multiplier_short):
    un_realized_profit = utils.get_unrealized_profit(response)
    if utils.in_profit(response):
        api.market_close_short(pair, quantity)
    else:
        if un_realized_profit > 0 and abs(utils.calculate_profit_margin(response)) > 5:  # Threshold
            additional_quantity = utils.calculate_new_quantity(quantity, multiplier_short)
            api.market_open_short(pair, additional_quantity)
        elif utils.stop_loss_triggered(response):
            api.market_close_short(pair, quantity)
            logger.info(f"Stop Loss triggered for {pair} short position. Closed position.")

def process_short_position(pair, response, quantity, hero, i):
    multiplier_short = config.multiplier_short[i]
    take_short_action(pair, response, quantity, hero, multiplier_short)

def handle_error(pair, e):
    if isinstance(e, (socket.timeout,
                      urllib3.exceptions.ProtocolError,
                      urllib3.exceptions.ReadTimeoutError,
                      requests.exceptions.ConnectionError,
                      requests.exceptions.ConnectTimeout,
                      requests.exceptions.ReadTimeout,
                      ConnectionResetError)):
        logger.error(f"Network error occurred in pair {pair}: {e}")
        # Add retry mechanism here if needed
    elif isinstance(e, BinanceAPIException):
        logger.error(f"Binance API error occurred in pair {pair}: {e}")
    elif isinstance(e, (KeyError, OSError)):
        logger.error(f"System error occurred in pair {pair}: {e}")
    else:
        logger.error(f"Unexpected error occurred in pair {pair}: {e}")

def run_trading_bot():
    pairs = deque(config.pair)
    while pairs:
        pair = pairs.popleft()
        leverage = config.leverage[pairs.index(pair)]
        quantity = config.quantity[pairs.index(pair)]
        try:
            response = api.position_information(pair)
            hero = combined.futures_hero(pair)
            make_money(pair, leverage, quantity)
            process_long_position(pair, response, quantity, hero, leverage, pairs.index(pair))
            process_short_position(pair, response, quantity, hero, pairs.index(pair))
            time.sleep(random.randint(1, 3))
            pairs.append(pair)
        except Exception as e:  # Catch all exceptions and handle them in handle_error function
            utils.handle_error(pair, e)
            pairs.append(pair)

if __name__ == "__main__":
    try:
        run_trading_bot()
    except KeyboardInterrupt:
        logger.info("\n\nAborted.\n")
