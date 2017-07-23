import numpy as np
import pandas as pd
import price_collection as pc
import news_collection as nc
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from datetime import date, timedelta


def collect_news():
    nc.init()
    root_dir = "/Users/liyuanqi/Google_Drive/UCLA_MSCS/Capstone/data"
    stock_symbol = "tsla"
    d1 = date(2016, 5, 1)  # start date
    d2 = date(2016, 6, 1)  # end date

    delta = d2 - d1         # timedelta

    for i in range(delta.days + 1):
        current_date = d1 + timedelta(days=i)
        print("[INFO] Now collecting news for {0} on: {1}-{2}-{3}".format(stock_symbol,current_date.year,current_date.month,current_date.day))
        urls = nc.get_ariticle_urls_for_date(stock_symbol=stock_symbol,year=current_date.year,month=current_date.month,day=current_date.day)
        for j in range(len(urls)):
            print("------ Downloading html from urls[{0}]: {1}".format(j,urls[j]))
            html = nc.get_html_with_url(urls[j])
            if(html==""):
                html = urls[j]
                nc.save_html_to_file(root_dir,stock_symbol,html,current_date.year,current_date.month,current_date.day,"empty")
            nc.save_html_to_file(root_dir,stock_symbol,html,current_date.year,current_date.month,current_date.day,j)
        nc.random_sleep(30,120)


def main():

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


collect_news()
