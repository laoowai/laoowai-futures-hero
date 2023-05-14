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
