import numpy as np
import pandas as pd
from yahoo_finance import Share
from datetime import date, timedelta
import re


price_csv_location = "/Users/liyuanqi/Google_Drive/UCLA_MSCS/Capstone/HistoricalQuotes.csv"

all_rows = None
date_to_price_map = {}
date_to_index_map = {}
sorted_dates = []

multiclass = False

def load_all_rows():
    global all_rows
    all_rows = pd.read_csv(price_csv_location, quotechar='"', skipinitialspace=True)
    all_rows = all_rows.as_matrix()
    np.delete(all_rows, 0, axis=0)


def build_date_to_price_map():
    global date_to_price_map
    global date_to_index_map
    global start_date
    global end_date
    global sorted_dates
    dates_text = [x[0] for x in all_rows]
    dates = []
    for i in range(0,len(dates_text)):
        converted_date = convert_string_to_date(dates_text[i])
        dates.append(converted_date)
    prices = [x[1] for x in all_rows]
    for i in range(0,len(dates)):
        date_to_price_map[dates[i]]=prices[i]
    sorted_dates = np.sort(dates)
    for i in range(0,len(sorted_dates)):
        date_to_index_map[sorted_dates[i]] = i


def init():
    load_all_rows()
    build_date_to_price_map()


def get_date_string(year,month,day):
    return "{0:d}/{1:02d}/{2:02d}".format(year,month,day)


def convert_string_to_date(string):
    m = re.match(r"(\d+)/(\d+)/(\d+)", string)
    return date(int(m.groups()[2])+2000,int(m.groups()[0]),int(m.groups()[1]))


def get_nth_date(n):
    return sorted_dates[n]


def get_trend_by_date(current_date, future_date):
    current_price = get_past_n_prices(current_date,1)[0]
    future_price = get_past_n_prices(future_date,1)[0]
    return future_price > current_price


def get_stock_price_by_date(date_string):
    return date_to_price_map.get(date_string)


def get_past_n_prices(date,n):
    prices=[]

    while(date not in sorted_dates):
        date = date - timedelta(days=1)
    date_index = date_to_index_map[date]
    if date_index-n<0:
        raise ValueError('date cannot be the oldest n dates')
    for i in range(date_index - n + 1, date_index + 1):
        prices.append(date_to_price_map[sorted_dates[i]])
    return prices


def get_past_n_trends(date,n):
    prices = get_past_n_prices(date,n+1)
    trends = []
    for i in range(len(prices)-1):
        trends.append(round((prices[i+1]-prices[i])*100.0/prices[i]))
        # trends.append(prices[i+1]>prices[i])
    return trends


def has_date(date):
    return date in sorted_dates

# init()
# print(get_trend_by_date(date(2015,12,31),date(2016,4,29)))

