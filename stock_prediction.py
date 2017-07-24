import numpy as np
import pandas as pd
import price_collection as pc
import news_collection as nc
from datetime import date, timedelta
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import RegexpTokenizer
from nltk.stem import LancasterStemmer
from sklearn.decomposition import TruncatedSVD


class CustomTokenizer(object):
    def __init__(self):
        self.regexp=RegexpTokenizer(r'\b([a-zA-Z]+)\b')
        self.stemmer = LancasterStemmer()
    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in self.regexp.tokenize(doc)]







def progression():

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


def classification():
    root_dir = "/Users/liyuanqi/Google_Drive/UCLA_MSCS/Capstone/data"
    stock_symbol = "tsla"    
    start_date = date(2016,5,15)
    end_date = date(2016,6,20)
    delta = end_date - start_date
    all_news = []
    all_trends = []
    for i in range(delta.days + 1):
        current_date = start_date + timedelta(days=i)
        if(pc.has_date(current_date)):
            news = nc.get_news_from_past_n_days(root_dir,stock_symbol,current_date,5)
            news = " ".join(news)
            trend = pc.get_trend_by_date(current_date)
            all_news.append(news)
            all_trends.append(trend)
    vectorizer = TfidfVectorizer(min_df=1, tokenizer=CustomTokenizer(), stop_words='english')
    
    data_train_vectorized = vectorizer.fit_transform(all_news)
    lsi = TruncatedSVD(n_components=50, random_state=42)
    data_train_matrix = lsi.fit_transform(data_train_vectorized)
    classifier = LogisticRegression()
    scores = cross_val_score(classifier, data_train_matrix, all_trends, cv=5,scoring='average_precision')
    print(scores)


def collect_news():

    root_dir = "/Users/liyuanqi/Google_Drive/UCLA_MSCS/Capstone/data"
    stock_symbol = "tsla"
    d1 = date(2016, 11, 21)  # start date
    d2 = date(2017, 7, 22)  # end date

    nc.collect_news(root_dir,stock_symbol,d1,d2)



def init():
    nc.init()
    pc.init()


def main():
    init()
    classification()

main()
