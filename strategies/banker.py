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
df['sell_signal'] = (df['predicted_up'].shift(1) == True) & (df['predicted_up'] == False)
