## Code
### webhook_launcher.py

```
import requests
import os, time, config

def telegram_bot_sendtext(bot_message):
    
    bot_token = ' '
    bot_chatID = ' '
    send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=html&text=' + bot_message
    response = requests.get(send_text)
    return response.json()
```
### utils.py

```
def calculate_percentage_difference(price1, price2):
    price1, price2 = max(price1, price2), min(price1, price2)
    return round((price1 - price2) / price1 * 100, 2)

def calculate_new_quantity(position_amt, multiplier):
    return round(float(position_amt) * multiplier, 2)

def calculate_new_price(entry_price, measure):
    return round((float(entry_price)) - (measure * (float(entry_price)) / 100), 2)

def calculate_breakeven_pnl(response, taker_fees):
    mark_price = float(response.get('markPrice'))
    position_amt = abs(float(response.get('positionAmt')))
    return (mark_price * position_amt * taker_fees) / 100

def in_profit(response, taker_fees):
    un_realized_pnl = float(response.get('unRealizedProfit'))
    breakeven_pnl = calculate_breakeven_pnl(response, taker_fees)
    return un_realized_pnl > breakeven_pnl

def in_profit_show(response, taker_fees):
    breakeven_pnl = calculate_breakeven_pnl(response, taker_fees)
    return round(breakeven_pnl,4)

def get_unrealized_profit(response):
    return float(response.get('unRealizedProfit'))

def calculate_profit_margin(response):
    un_realized_profit = get_unrealized_profit(response)
    mark_price = float(response.get('markPrice'))
    position_amt = abs(float(response.get('positionAmt')))
    revenue = mark_price * position_amt
    return (un_realized_profit / revenue) * 100 if revenue != 0 else 0

def stop_loss_triggered(response, stop_loss_percentage):
    return calculate_profit_margin(response) < -stop_loss_percentage
```

### run.py

```
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
```

### config.py

```
live_trade = True

coin     = ["BNB"]
quantity = [0.75]
multiplier_short = [0.75]  # Multiplier for short position
multiplier_long = [1.2]  # Multiplier for long position
stop_loss_percentage = [3]

leverage, pair = [], []

for i in range(len(coin)):
    pair.append(coin[i] + "BUSD")
    if   coin[i] == "BTC": leverage.append(2)
    elif coin[i] == "ETH": leverage.append(2)
    else: leverage.append(2)

    print("Pair Name        :   " + pair[i])
    print("Trade Quantity   :   " + str(quantity[i]) + " " + coin[i])
    print("Leverage         :   " + str(leverage[i]))
    print()

# long
add_long_measure = 35
new_quantity_long_multiplier = 0.30
takeprofitlong_usd = 0.5
liquidationpricemarklong_percent = 5

# change long quantity
quantity_long_sep = 0.99
quantity_short_sep = 0.25

# short
add_short_measure = 35
new_quantity_short_multiplier = 0.30
takeprofitshort_usd = 0.5
liquidationpricemarkshort_percent = 5

# long position / short position
# "add_long_measure" is a % between entry and current positions, when exceed % add "new_quantity_long"
# "new_quantity_long_multiplier" is a multiplier for every next order, if multiplier smaller than every next
# order will be smaller than order before
# "liquidationpricemarkshort_percent" % before liqudation add "quantity", better to keep around 5-10%
```

### backtest.py

```
import config
import strategies.combined
import strategies.ichimoku
import strategies.volume
import strategies.william_fractal
from datetime import datetime

fees = 0.1
choose_your_fighter = strategies.volume

def backtest():
    all_pairs = 0
    for i in range(len(config.pair)):
        print(config.pair[i])
        leverage = config.leverage[i]
        hero = choose_your_fighter.futures_hero(config.pair[i])
        # print(hero)

        print("Start Time Since " + str(datetime.fromtimestamp(hero["timestamp"].iloc[0]/1000)))
        long_result = round(check_PNL(hero, leverage, "_LONG"), 2)
        short_reult = round(check_PNL(hero, leverage, "SHORT"), 2)
        overall_result = round(long_result + short_reult, 2)
        all_pairs = round(all_pairs + overall_result, 2)

        print("PNL for _BOTH Positions: " + str(overall_result) + "%\n")
    print("ALL PAIRS PNL : " + str(all_pairs) + "%\n")

def check_PNL(hero, leverage, positionSide):
    position = False
    total_pnl, total_trades, liquidations = 0, 0, 0
    wintrade, losetrade = 0, 0

    if positionSide == "_LONG":
        open_position = "GO_LONG"
        exit_position = "EXIT_LONG"
        liq_indicator = "low"

    elif positionSide == "SHORT":
        open_position = "GO_SHORT"
        exit_position = "EXIT_SHORT"
        liq_indicator = "high"

    for i in range(len(hero)):
        if not position:
            if hero[open_position].iloc[i]:
                entry_price = hero['close'].iloc[i]
                position = True
        else:
            liquidated = (hero[liq_indicator].iloc[i] - entry_price) / entry_price * 100 * leverage < -80
            unrealizedPNL = (hero['close'].iloc[i] - entry_price) / entry_price * 100 * leverage
            breakeven_PNL = fees * leverage

            if (hero[exit_position].iloc[i] and unrealizedPNL > breakeven_PNL) or liquidated:
                if liquidated:
                    realized_pnl = -100
                    liquidations = liquidations + 1
                else: realized_pnl = unrealizedPNL - breakeven_PNL

                if realized_pnl > 0: wintrade = wintrade + 1
                else: losetrade = losetrade + 1

                total_trades = total_trades + 1
                total_pnl = total_pnl + realized_pnl
                position = False

    if total_pnl != 0:
        print("PNL for " + positionSide + " Positions: " + str(round(total_pnl, 2)) + "%")
        print("Total  Executed  Trades: " + str(round(total_trades, 2)))
        print("Total Liquidated Trades: " + str(round(liquidations)))
        print("_Win Trades: " + str(wintrade))
        print("Lose Trades: " + str(losetrade))
        if (wintrade + losetrade > 1):
            winrate = round(wintrade / (wintrade + losetrade) * 100)
            print("Winrate : " + str(winrate) + " %")
        print()

    return round(total_pnl, 2)

backtest()
```

### api_binance.py

```
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
```

### strategies/banker.py

```
# Import
import ccxt
import pandas as pd
import matplotlib.pyplot as plt
import ta

from ta import add_all_ta_features
from ta.utils import dropna
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Initialize Binance API
exchange = ccxt.binance()

# Fetch the market data
ohlcv = exchange.fetch_ohlcv('BNB/USDT', '15m')

# Convert to DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Plot
plt.plot(df['timestamp'], df['close'])
plt.show()

# Inputs
ema_threshold = 14
ma_length1 = 3
ma_length2 = 6
rsi_period = 14
sma_period = 3
typical_price_period = 15
stochastic_period = 18
stochastic_smoothing1 = 9
stochastic_smoothing2 = 3
resistance_period = 60
ema_close_period = 14
over_bought_level = 98
over_sold_level = 2
neutral_level = 50
rsi_over_bought_level = 70
rsi_over_sold_level = 30

# Custom Functions
def fill_na(values):
    return values.fillna(method='ffill')

def smoothed_average(src, length):
    return src.rolling(length).mean()

# Assuming that candlestick, EMA, heikin_ashi, MACD are functions that return updated dataframes.
# Replace these with your actual functions.
from modules import candlestick, EMA, heikin_ashi, MACD
df = candlestick(df)
df = EMA(df, ema_threshold)
df = heikin_ashi(df)
df = MACD(df)

# Calculate RSI
df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], n=rsi_period).rsi()

# Calculations
df['basic_ma'] = df['close'].rolling(sma_period).mean()
df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
df['highest_price'] = df['typical_price'].rolling(typical_price_period).max()
df['lowest_price'] = df['typical_price'].rolling(typical_price_period).min()
df['price_range'] = df['highest_price'] - df['lowest_price']
df['price_percentage'] = ((df['typical_price'] - df['lowest_price']) / df['price_range']).rolling(2).mean() * 100
df['percent_change'] = (df['close'] - df['close'].shift(1).fillna(method='ffill')) / df['close'].shift(1).fillna(method='ffill') * 100
df['stochastic'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], n=stochastic_period, d_n=stochastic_smoothing1).stoch_signal()
df['ema_base'] = df['close'].ewm(span=ema_close_period).adjust=False.mean()
df['ema1'] = df['ema_base'].ewm(span=ma_length1, adjust=False).mean()
df['ema2'] = df['ema1'].ewm(span=ma_length1, adjust=False).mean()
df['ema3'] = df['ema2'].ewm(span=ma_length1, adjust=False).mean()
df['ema4'] = df['ema3'].ewm(span=ma_length1, adjust=False).mean()
df['smoothed_ema'] = df['ema4'].ewm(span=ma_length2, adjust=False).mean()
df['resistance'] = df['stochastic'].rolling(resistance_period).max()
df['support'] = df['stochastic'].rolling(resistance_period).min()

df['long_condition'] = ((df['price_percentage'].shift(1) < over_sold_level) &
                      (df['price_percentage'] > over_sold_level) &
                      (df['stochastic'].rolling(3).min() == df['support']))
df['short_condition'] = ((df['price_percentage'].shift(1) > over_bought_level) &
                       (df['price_percentage'] < over_bought_level) &
                       (df['stochastic'].rolling(3).max() == df['resistance']))
df['RSI_buy'] = df['long_condition'] & (df['rsi'] < rsi_over_sold_level)
df['RSI_sell'] = df['short_condition'] & (df['rsi'] > rsi_over_bought_level)

# Define labels
df['up'] = df['close'].shift(-1) > df['close']

# Define features
features = ['rsi', 'basic_ma', 'typical_price', 'highest_price', 'lowest_price', 'price_range', 'price_percentage',
            'percent_change', 'stochastic', 'ema_base', 'ema1', 'ema2', 'ema3', 'ema4', 'smoothed_ema', 'resistance', 'support']

# Drop rows with missing values
df = df.dropna()

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df['up'], test_size=0.2, shuffle=False)

# Create and train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Generate predictions
df['predicted_up'] = clf.predict(df[features])

# Generate trading signals based on predictions
df['buy_signal'] = (df['predicted_up'].shift(1) == False) & (df['predicted_up'] == True)
df['sell_signal'] = (df['predicted_up'].shift(1) == True) & (df['predicted_up'] == False
```

### modules/candlestick.py

```
import ccxt
import pandas

query = 1000
ccxt_client = ccxt.binance()
tohlcv_colume = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
def get_klines(pair, interval):
    return pandas.DataFrame(ccxt_client.fetch_ohlcv(pair, interval , limit=query), columns=tohlcv_colume)

def candlestick(klines):
    candlestick_df = klines # make a new DataFrame called candlestick_df

    # Temporary previous column
    candlestick_df["high_s1"] = klines['high'].shift(1)
    candlestick_df["high_s2"] = klines['high'].shift(2)
    candlestick_df["low_s1"]  = klines['low'].shift(1)
    candlestick_df["low_s2"]  = klines['low'].shift(2)

    # Compute candlestick details
    candlestick_df["color"]  = candlestick_df.apply(candle_color, axis=1)
    candlestick_df["upper"]  = candlestick_df.apply(upper_wick, axis=1)
    candlestick_df["lower"]  = candlestick_df.apply(lower_wick, axis=1)
    candlestick_df["body"]   = abs(candlestick_df['open'] - candlestick_df['close'])
    candlestick_df["strong"] = candlestick_df.apply(strong_candle, axis=1)
    candlestick_df["volumeAvg"] = sum(klines["volume"]) / query / 3

    clean = candlestick_df[["timestamp", "open", "high", "low", "close", "volume", "volumeAvg", "color", "strong"]].copy()
    return clean

# ==========================================================================================================================================================================
#                                                           PANDAS CONDITIONS
# ==========================================================================================================================================================================

def candle_color(candle):
    if candle['close'] > candle['open']: return "GREEN"
    elif candle['close'] < candle['open']: return "RED"
    else: return "INDECISIVE"

def upper_wick(candle):
    if candle['color'] == "GREEN": return candle['high'] - candle['close']
    elif candle['color'] == "RED": return candle['high'] - candle['open']
    else: return (candle['high'] - candle['open'] + candle['high'] - candle['close']) / 2

def lower_wick(candle):
    if candle['color'] == "GREEN": return candle['open'] - candle['low']
    elif candle['color'] == "RED": return candle['close'] - candle['low']
    else: return (candle['open'] - candle['low'] + candle['close'] - candle['low']) / 2

def strong_candle(candle):
    if candle["color"] == "GREEN": return True if candle['close'] > candle['high_s1'] and candle['close'] > candle['high_s2'] else False
    elif candle["color"] == "RED": return True if candle['close'] < candle['low_s1'] and candle['close'] < candle['low_s2'] else False
    else: return False

def test_module():
    klines = get_klines("BTCUSDT", "1h")
    processed_candle = candlestick(klines)
    print("\ncandlestick.candlestick(klines)")
    print(processed_candle)

# test_module()
```

### modules/EMA.py

```
def apply_EMA(dataset, EMA_threshold):
    dataset[str(EMA_threshold) + 'EMA'] = dataset['close'].ewm(span=EMA_threshold).mean()
    clean = dataset[["timestamp", str(EMA_threshold) + "EMA"]].copy()
    return clean
```

### modules/MACD.py

```
MACD_threshold = 50

def apply_MACD(dataset):
    dataset['12_EMA'] = dataset['close'].ewm(span=12).mean()
    dataset['26_EMA'] = dataset['close'].ewm(span=26).mean()
    dataset['MACD'] = dataset['12_EMA'] - dataset['26_EMA']
    dataset['Signal'] = dataset['MACD'].ewm(span=9).mean()
    dataset['Histogram'] = dataset['MACD'] - dataset['Signal']
    dataset['MACD_long'] = dataset.apply(long_condition, axis=1)
    dataset['MACD_short'] = dataset.apply(short_condition, axis=1)

    clean = dataset[["timestamp", "MACD_long", "MACD_short"]].copy()
    return clean

def long_condition(dataset):
    if  dataset['Signal'] < MACD_threshold and \
        dataset['Histogram'] > 0 : return True 
    else: return False

def short_condition(dataset):
    if  dataset['Signal'] > -MACD_threshold and \
        dataset['Histogram'] < 0 : return True
    else: return False

def test_module():
    import candlestick, heikin_ashi
    klines = candlestick.get_klines("BTCUSDT", "1h")
    # heikin = heikin_ashi.heikin_ashi(klines)
    applyMACD = apply_MACD(klines)
    print("\nMACD.apply_MACD(klines)")
    print(applyMACD)

# test_module()
```

### modules/heikin_ashi.py

```
import pandas

def heikin_ashi(klines):
    heikin_ashi_df = pandas.DataFrame(index=klines.index.values, columns=['open', 'high', 'low', 'close'])
    heikin_ashi_df['close'] = (klines['open'] + klines['high'] + klines['low'] + klines['close']) / 4

    for i in range(len(klines)):
        if i == 0: heikin_ashi_df.iat[0, 0] = klines['open'].iloc[0]
        else: heikin_ashi_df.iat[i, 0] = (heikin_ashi_df.iat[i-1, 0] + heikin_ashi_df.iat[i-1, 3]) / 2

    heikin_ashi_df['high'] = heikin_ashi_df.loc[:, ['open', 'close']].join(klines['high']).max(axis=1)
    heikin_ashi_df['low']  = heikin_ashi_df.loc[:, ['open', 'close']].join(klines['low']).min(axis=1)
    heikin_ashi_df["color"] = heikin_ashi_df.apply(color, axis=1)
    heikin_ashi_df.insert(0,'timestamp', klines['timestamp'])
    heikin_ashi_df["volume"] = klines["volume"]

    # Use Temporary Column to Identify Strength
    heikin_ashi_df["upper"] = heikin_ashi_df.apply(upper_wick, axis=1)
    heikin_ashi_df["lower"] = heikin_ashi_df.apply(lower_wick, axis=1)
    heikin_ashi_df["body"]  = abs(heikin_ashi_df['open'] - heikin_ashi_df['close'])
    heikin_ashi_df["body_s1"] = heikin_ashi_df['body'].shift(1)
    heikin_ashi_df["body_s2"] = heikin_ashi_df['body'].shift(2)
    heikin_ashi_df["indecisive"] = heikin_ashi_df.apply(absolute_indecisive, axis=1)
    heikin_ashi_df["candle"] = heikin_ashi_df.apply(valid_candle, axis=1)

    clean = heikin_ashi_df[["timestamp", "open", "high", "low", "close", "volume", "color", "candle", "indecisive"]].copy()
    return clean

# ==========================================================================================================================================================================
#                                                           PANDAS CONDITIONS
# ==========================================================================================================================================================================

def color(HA):
    if   HA['open'] < HA['close']: return "GREEN"
    elif HA['open'] > HA['close']: return "RED"
    else: return "INDECISIVE"

def upper_wick(HA):
    if HA['color'] == "GREEN": return HA['high'] - HA['close']
    elif HA['color'] == "RED": return HA['high'] - HA['open']
    else: return (HA['high'] - HA['open'] + HA['high'] - HA['close']) / 2

def lower_wick(HA):
    if HA['color'] == "GREEN": return  HA['open'] - HA['low']
    elif HA['color'] == "RED": return HA['close'] - HA['low']
    else: return (HA['open'] - HA['low'] + HA['close'] - HA['low']) / 2

def absolute_indecisive(HA):
    if HA['body'] * 2 < HA['upper'] and HA['body'] * 2 < HA['lower'] : return True
    else: return False

def valid_candle(HA):
    if not HA['indecisive']:
        if HA['color'] == "GREEN": return "GREEN"
        elif HA['color'] == "RED": return "RED"
    else: return "INDECISIVE"

def test_module():
    import candlestick
    klines = candlestick.get_klines("BTCUSDT", "1h")
    processed_heikin_ashi = heikin_ashi(klines)
    print("\nheikin_ashi.heikin_ashi")
    print(processed_heikin_ashi)

# test_module()
```

### utils.py

```
```

### utils.py

```
```

### utils.py

```
```

### utils.py

```
```

### utils.py

```
```

### utils.py

```
```


# Project Update:
I am currently not able to give this project enough time to fix the current issues or add new features.  
I am busy with some other projects. But I do plan to fix all the issues and add some new features.  
So the maintenance is temporarily on hold and this project is not dead.

## TABLE OF CONTENTS

1. [FUTURES-HERO](#futures_hero)
2. [DISCLAIMER](#hello_disclaimer)
3. [HOW-IT-WORKS](#how_it_works)
4. [HOW-TO-USE](#how_to_use)
    1. [ENVIRONMENT SETUP](#environment_setup)
    2. [PIP3 REQUIREMENTS](#pip3_requirements)
    3. [CONFIGURATIONS](#configurations)
    4. [RUN](#run)
5. [SCREENSHOTS](#hello_screenshots)
    - [SAMPLE-OUTPUT](#sample_output)
6. [JOIN-MY-DISCORD](#discord)
    - [QUICK ACCESS TO THE DARK DIMENSION](https://discord.gg/6J2mXvYsFB)
    - Please email or create an issue if the invitation link does not work  

<a name="futures_hero"></a>
## FUTURES-HERO
Leverage Trading Automation on Binance Futures.  
This is a `Set and Forget` script, means you need to keep it running 24/7 and forget about it.  
**The bot is stable in current version therefore no new changes until new bugs been spotted.**  
**I do not use this bot personally, however, I do use the other 2 bots:**  
- https://github.com/zyairelai/buy-low-sell-high
- https://github.com/zyairelai/long-term-low-leverage

<a name="hello_disclaimer"></a>
## DISCLAIMER
This automation software is implemented base on my PERSONAL MANUAL TRADING STRATEGY.  
However not all my manual trading strategies are completely transformed into code.  
For example, common sense, 6th sense, knowing when to stop trading are **NOT** the part I could do in this code.  

**LEVERAGE TRADING IS A HIGH RISK GAME.**  
**PLEASE MANAGE YOUR RISK LEVEL BEFORE USING MY SCRIPT.**

<a name="how_it_works"></a>
## HOW-IT-WORKS
In short, this code takes these few conditions into considerations:  
4-hour timeframe, 1-hour timeframe and 1-minute timeframe

1. It checks the overall main direction, 4-hour.  
   Let's say the newest current 4-hour candle is `GREEN`.  

2. Then it checks the 1-hour candle for confirmation.  
   Let's say the newest current 1-hour candle is matched with 4-hour candle `GREEN`.  

3. Since both main direction and confirmation are `GREEN`, now it will find an entry  
   (The part which I am not able to implement into code). 

4. The entry will be on the 1-minute chart.  
   In the case above, it will look for a `LONG` position entry.

<a name="how_to_use"></a>
## HOW-TO-USE
<a name="environment_setup"></a>
### 1. ENVIRONMENT SETUP
Paste the following into your Default Shell
```
export BINANCE_KEY="your_binance_api_key"
export BINANCE_SECRET="your_binance_secret_key"
```

Or as an ALTERNATIVE, you can change `line 7-9` in `binance_futures_api.py` to following: 
```
api_key     = "your_binance_api_key"
api_secret  = "your_binance_secret_key"
client      = Client(api_key, api_secret)
```
Don't forget the `" "` symbol to make your API key into `STRING` type!  

**I WILL NO LONGER ANSWER QUESTION REGARDING TO THIS ERROR:**
```
AttributeError: 'NoneType' object has no attribute 'encode'
``` 
**QUICK GOOGLE SEARCH or FIX YOUR API KEY**  
**DO NOT SPAM MY EMAIL AND DISTURB MY PEACEFUL LIFE LOL**

<a name="pip3_requirements"></a>
### 2. PIP3 REQUIREMENTS
```
Workig with Python 3.6.8**
sudo apt-get install python3
```
To install all the dependencies requirements in one line:
```
pip3 install -r requirements.txt
```
Or if you prefer to install these libraries one by one:
```
pip3 install ccxt
pip3 install numpy
pip3 install pandas
pip3 install requests
pip3 install termcolor
pip3 install python-binance
pip3 install cryptography==3.4.6

pip3 install pyOpenSSL

```

<a name="configurations"></a>
### 3. CONFIGURATIONS
Before running, maybe you want to see how the output looks like.  
The settings can be configured in `config.py`.

| Variables           | Description                                                                                            |
| --------------------| -------------------------------------------------------------------------------------------------------|
| `live_trade`        |`True` to place actual order <br /> `False` to see sample output                                        |
| `coin`              | You can put your coin list here, to add more it will be ["BTC", "ETH"]                                 |
| `quantity`          | Trade amount. You can run `util_cal_tradeAmt.py` to check the trade amount                             |

The following code illustrate how you add more pairs to trade with:  
```
coin = ["BTC", "ETH"]
quantity = [0.001, 0.01]
```
**IMPORTANT NOTE:**  
- Check your minimum trade size here https://www.binance.com/en/futures/trading-rules/perpetual

<a name="run"></a>
### 4. RUN
You can select your strategy in the `strategies` folder.  

Just replace the `strategy.py` file with any strategy in that folder and you are good to go.  

The default strategy is `strategy_hybrid.py`.  

Now if you are all ready, set `live_trade = True` and ...

Let's make the magic happens!
```
python3 run.py
```

**Make sure you are having Python 3.8 as default python, else you might need to specify your path with python3.8**

<a name="hello_screenshots"></a>
## SCREENSHOTS

<a name="sample_output"></a>
### SAMPLE OUTPUT
<p align="center">
  <img src="screenshots/sample_output.png">
</p>

<a name="discord"></a>
## [JOIN MY DISCORD - QUICK ACCESS TO THE DARK DIMENSION](https://discord.gg/6J2mXvYsFB)
### Please email or create an issue if the invitation link does not work  
