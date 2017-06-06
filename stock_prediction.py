import numpy as np
import pandas as pd
import price_collection as pc
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score


pc.init()

n=5
past_price_matrix = []
actual_prices = []
for i in range(n,len(pc.sorted_dates)):
    past_price_matrix.append(
        pc.get_past_n_prices(i,n))
    actual_prices.append(pc.get_stock_price_by_date(pc.sorted_dates[i]))
# print(past_price_matrix)


model = MLPRegressor(activation="logistic", solver='lbfgs')

# model.fit(past_price_matrix, actual_prices)

print(cross_val_score(model, past_price_matrix, actual_prices, cv=5))
