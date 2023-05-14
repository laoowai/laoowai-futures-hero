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
