import numpy as np
import pandas as pd
from yahoo_finance import Share
from datetime import date, timedelta
import re


price_csv_location = "/Users/liyuanqi/Google_Drive/UCLA_MSCS/Capstone/HistoricalQuotes.csv"

all_rows = None
date_to_price_map = {}
date_to_trend_map = {}
date_to_index_map = {}
sorted_dates = []


def load_all_rows():
    global all_rows
    all_rows = pd.read_csv(price_csv_location, quotechar='"', skipinitialspace=True)
    all_rows = all_rows.as_matrix()
    np.delete(all_rows, 0, axis=0)


def build_date_to_price_map():
    global date_to_price_map
    global date_to_index_map
    global date_to_trend_map
    global start_date
    global end_date
    global sorted_dates
    dates_text = [x[0] for x in all_rows]
    dates = []
    for i in range(0,len(dates_text)):
        dates.append(convert_string_to_date(dates_text[i]))
    prices = [x[1] for x in all_rows]
    for i in range(1,len(dates)):
        date_to_trend_map[dates[i]]=(prices[i]<prices[i-1])
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
    return date(int(m.groups()[0]),int(m.groups()[1]),int(m.groups()[2]))


def get_nth_date(n):
    return sorted_dates[n]


def get_trend_by_date(date):
    return date_to_trend_map[date]


# return price for the date, if no price, return -1
def get_stock_price_by_date(date_string):
    return date_to_price_map.get(date_string, -1)


def get_past_n_prices(year,month,day,n):
    prices=[]
    date_index = date_to_index_map[get_date_string(year,month,day)]
    if date_index-n<0:
        raise ValueError('date cannot be the oldest n dates')
    for i in range(date_index - n, date_index):
        prices.append(date_to_price_map[sorted_dates[i]])
    return prices


def get_past_n_prices(index, n):
    prices=[]
    date_index = index
    if date_index-n<0:
        raise ValueError('date cannot be the oldest n dates')
    for i in range(date_index - n, date_index):
        prices.append(date_to_price_map[sorted_dates[i]])
    return prices


def has_date(date):
    return date in sorted_dates
