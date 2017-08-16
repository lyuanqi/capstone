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


root_dir = "/Users/liyuanqi/Google_Drive/UCLA_MSCS/Capstone/data"
stock_symbol = "tsla"    


def get_date_duration_string(start_date,end_date):
    date_duration_string = "{0}{1}{2}-{3}{4}{5}".format(
        start_date.year,
        start_date.month,
        start_date.day,
        end_date.year,
        end_date.month,
        end_date.day)
    return date_duration_string


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
        actual_prices.append((pc.sorted_dates[i]))
    # print(past_price_matrix)


    model = MLPRegressor(activation="logistic", solver='lbfgs')

    # model.fit(past_price_matrix, actual_prices)

    print(cross_val_score(model, past_price_matrix, actual_prices, cv=5))


def prepare_data(past_n,start_date,end_date,keyword_num,term_offset,cv):
    print("[data prep] preparing training data...")
    trends_data_file_name = 'all_trends_{0}'.format(get_date_duration_string(start_date,end_date)) 
    vectorized_data_file_name = "n{0}_vectorized_data_{1}".format(past_n,get_date_duration_string(start_date,end_date))
    past_n_trends_matrix_file_name = "past_{0}_trends_matrix_{1}".format(past_n,get_date_duration_string(start_date,end_date))
    all_news = []
    all_trends = []
    past_n_trends_matrix = []
    delta = end_date - start_date
    for i in range(delta.days + 1):
        current_date = start_date + timedelta(days=i)
        if(pc.has_date(current_date)):
            news = nc.get_news_from_past_n_days(root_dir,stock_symbol,current_date,past_n)
            news = " ".join(news)
            trend = pc.get_trend_by_date(current_date + timedelta(days=term_offset))
            all_news.append(news)
            all_trends.append(trend)
            trends = pc.get_past_n_trends(current_date,past_n)
            past_n_trends_matrix.append(trends)

    vectorizer = TfidfVectorizer(min_df=1, tokenizer=CustomTokenizer(), stop_words='english')
    data_train_vectorized = vectorizer.fit_transform(all_news)

    features = vectorizer.get_feature_names()
    features = np.array(features)
    data_train_vectorized = data_train_vectorized.toarray()
    top_index_words = np.argsort(data_train_vectorized[0])[::-1]
    print("Top 10 key words:")
    print(features[top_index_words[0:10].tolist()])

    np.save(vectorized_data_file_name,data_train_vectorized)
    np.save(trends_data_file_name,all_trends)
    np.save(past_n_trends_matrix_file_name,past_n_trends_matrix)


def classification(past_n,start_date,end_date,keyword_num,term_offset,cv):
    print("[classification] loading training data from .npy files...")
    vectorized_data_file_name = "n{0}_vectorized_data_{1}.npy".format(past_n,get_date_duration_string(start_date,end_date))
    trends_data_file_name = 'all_trends_{0}.npy'.format(get_date_duration_string(start_date,end_date)) 
    past_n_trends_matrix_file_name = "past_{0}_trends_matrix_{1}.npy".format(past_n,get_date_duration_string(start_date,end_date))
    data_train_vectorized = np.load(vectorized_data_file_name)
    all_trends = np.load(trends_data_file_name)
    past_n_trends_matrix = np.load(past_n_trends_matrix_file_name)

    # print("Trend Ratio: {0} Increases (True) : {1} Decreases (False)").format(np.sum(all_trends), len(all_trends) - np.sum(all_trends))
    lsi = TruncatedSVD(n_components=keyword_num, random_state=42)
    data_train_matrix = lsi.fit_transform(data_train_vectorized)
    data_train_matrix_with_past_price = np.c_[data_train_matrix, past_n_trends_matrix]
    classifier = LogisticRegression()
    print("[classification] training and predicting...")
    scores = cross_val_score(classifier, data_train_matrix_with_past_price, all_trends, cv=cv,scoring='average_precision')
    print(scores)


def collect_news():
    amzn_root_dir = "/Users/liyuanqi/Google_Drive/UCLA_MSCS/Capstone/data/amzn"
    stock_symbol = "amzn"
    d1 = date(2017, 6, 6)  # start date
    d2 = date(2017, 7, 1)  # end date
    nc.collect_news(amzn_root_dir,stock_symbol,d1,d2)


def init():
    nc.init()
    pc.init()


def main():
    init()
    cv = 2
    past_n = 5
    start_date = date(2016,7,1)
    end_date = date(2016,7,30)
    keyword_num = 10
    term_offset = 10
    prepare_data(past_n,start_date,end_date,keyword_num,term_offset,cv)
    classification(past_n,start_date,end_date,keyword_num,term_offset,cv)


init()
main()
# collect_news()



# todo:
# build progression into classification or vice versa
# params tuning
# read more relevant papers
# train with other models
# dedensify dates
