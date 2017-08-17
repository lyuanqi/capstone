import os.path
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


data_root_dir = "/Users/liyuanqi/Google_Drive/UCLA_MSCS/Capstone/data"
stock_symbol = "tsla"    


def file_exists(file_path):
    return os.path.isfile(file_path) 


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
    trends_data_file_name = '{0}_all_trends_{1}'.format(data_root_dir + "/" + stock_symbol,get_date_duration_string(start_date,end_date)) 
    vectorized_data_file_name = "{0}_n{1}_vectorized_data_{2}".format(data_root_dir + "/" + stock_symbol,past_n,get_date_duration_string(start_date,end_date))
    past_n_trends_matrix_file_name = "{0}_past_{1}_trends_matrix_{2}".format(data_root_dir + "/" + stock_symbol,past_n,get_date_duration_string(start_date,end_date))
    all_news = []
    all_trends = []
    past_n_trends_matrix = []
    delta = end_date - start_date
    for i in range(delta.days + 1):
        current_date = start_date + timedelta(days=i)
        if(pc.has_date(current_date)):
            news = nc.get_news_from_past_n_days(data_root_dir + "/" + stock_symbol,stock_symbol,current_date,past_n)
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
    trends_data_file_name = '{0}_all_trends_{1}.npy'.format(data_root_dir + "/" + stock_symbol,get_date_duration_string(start_date,end_date)) 
    vectorized_data_file_name = "{0}_n{1}_vectorized_data_{2}.npy".format(data_root_dir + "/" + stock_symbol,past_n,get_date_duration_string(start_date,end_date))
    past_n_trends_matrix_file_name = "{0}_past_{1}_trends_matrix_{2}.npy".format(data_root_dir + "/" + stock_symbol,past_n,get_date_duration_string(start_date,end_date))
    
    all_data_files_exist = file_exists(trends_data_file_name) and file_exists(vectorized_data_file_name) and file_exists(past_n_trends_matrix_file_name)
    if(not all_data_files_exist):
        prepare_data(past_n,start_date,end_date,keyword_num,term_offset,cv)

    data_train_vectorized = np.load(vectorized_data_file_name)
    all_trends = np.load(trends_data_file_name)
    past_n_trends_matrix = np.load(past_n_trends_matrix_file_name)

    # print("Trend Ratio: {0} Increases (True) : {1} Decreases (False)").format(np.sum(all_trends), len(all_trends) - np.sum(all_trends))
    lsi = TruncatedSVD(n_components=keyword_num, random_state=42)
    data_train_matrix = lsi.fit_transform(data_train_vectorized)
    data_train_matrix_with_past_price = np.c_[data_train_matrix, past_n_trends_matrix]
    classifier = LogisticRegression()
    print("[classification] training and predicting with past_n={0},keyword_num={1},term_offset={2}...".format(past_n,keyword_num,term_offset))
    scores = cross_val_score(classifier, data_train_matrix_with_past_price, all_trends, cv=cv,scoring='average_precision')
    return scores


def train_with_params(start_date,end_date,cv,past_n_range,keyword_num_range,term_offset_range):
    all_results = []
    for past_n in past_n_range:
        for keyword_num in keyword_num_range:
            for term_offset in term_offset_range:
                params = {}
                params['past_n'] = past_n
                params['keyword_num'] = keyword_num
                params['term_offset'] = term_offset
                scores = classification(past_n,start_date,end_date,keyword_num,term_offset,cv)
                result = {}
                result['params'] = params
                result['scores'] = scores
                all_results.append(result)
    for i in range(len(all_results)):
        result = all_results[i]
        print(result['params'])
        print(result['scores'])


def collect_news():
    amzn_data_root_dir = "/Users/liyuanqi/Google_Drive/UCLA_MSCS/Capstone/data/amzn"
    stock_symbol = "amzn"
    d1 = date(2017, 6, 6)  # start date
    d2 = date(2017, 7, 1)  # end date
    nc.collect_news(amzn_data_root_dir,stock_symbol,d1,d2)


def init():
    nc.init()
    pc.init()


def main():
    init()
    cv = 2
    start_date = date(2016,7,1)
    end_date = date(2016,7,30)
    
    # test ranges
    # past_n_range = np.arange(3,5,1)
    # keyword_num_range = np.arange(10,20,5)
    # term_offset_range = np.arange(1,3,1)

    #real ranges
    past_n_range = np.arange(1,61,1)
    keyword_num_range = np.arange(10,150,10)
    term_offset_range = np.arange(1,61,1)

    train_with_params(start_date,end_date,cv,past_n_range,keyword_num_range,term_offset_range)
    # classification(past_n,start_date,end_date,keyword_num,term_offset,cv)


init()
main()
# collect_news()



# todo:
# build progression into classification or vice versa
# params tuning
# read more relevant papers
# train with other models
# dedensify dates
