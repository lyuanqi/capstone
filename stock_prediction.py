import os.path
import numpy as np
import pandas as pd
import price_collection as pc
import news_collection as nc
import matplotlib.pyplot as plt
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
                print_and_write_to_file("** | past_n={0} | keyword_num={1} | term_offset={2} | AVG_SCORE = {3}"
                    .format(result['params']['past_n'],result['params']['keyword_num'],result['params']['term_offset'],np.mean(result['scores'])))
                print_and_write_to_file("** scores: [{0}]".format(', '.join([str(x) for x in result['scores']])))
                print_and_write_to_file("-------------------------------------------------------------------------")
            
    # print_and_write_to_file("-------------------------------------------------------------------------")
    # for i in range(len(all_results)):
    #     result = all_results[i]
    #     print_and_write_to_file("** | past_n={0} | keyword_num={1} | term_offset={2} | AVG_SCORE = {3}"
    #         .format(result['params']['past_n'],result['params']['keyword_num'],result['params']['term_offset'],np.mean(result['scores'])))
    #     print_and_write_to_file("** scores: [{0}]".format(', '.join([str(x) for x in result['scores']])))
    #     print_and_write_to_file("-------------------------------------------------------------------------")
    
    np.save('all_results',all_results)


def plot_for_varying_x(x_name,x_range,y_range):
    plt.figure()
    axes = plt.gca()
    plt.xlabel(x_name)
    plt.ylabel("AVG Accuracy")
    plt.title("My Title")
    plt.plot(x_range, y_range, color = 'r', ls='-', marker='o')
    plt.savefig(x_name + "_accuracy_plot")
    plt.close()


def print_and_write_to_file(string):
    f = open("all_results.txt", "a")
    f.write(string + "\n")
    print(string)
    f.close()


def collect_news():
    amzn_data_root_dir = "/Users/liyuanqi/Google_Drive/UCLA_MSCS/Capstone/data/tsla"
    stock_symbol = "tsla"
    d1 = date(2016, 1, 1)  # start date
    d2 = date(2016, 4, 30)  # end date
    nc.collect_news(amzn_data_root_dir,stock_symbol,d1,d2)


def init():
    nc.init()
    pc.init()
    try:
        os.remove('all_results.txt')
    except OSError:
        pass


def main():
    init()
    cv = 2
    test = False
    # test_dates
    start_date = date(2016,8,1)
    end_date = date(2017,4,1)
    
    # test ranges
    past_n_range = [5,11,15]
    keyword_num_range = [50]
    term_offset_range = [1,5,10]
    if(not test):
        print("[main] using real ranges...")
        cv = 5
        # real dates
        start_date = date(2016,4,1)
        end_date = date(2017,4,1)
        # real ranges
        past_n_range = [5,10,15,20,25,30]
        # keyword_num_range = np.arange(50,160,30)
        # term_offset_range = [1,5,10,15,20,25,30]

        keyword_num_range = [10]
        term_offset_range = [1]

    train_with_params(start_date,end_date,cv,past_n_range,keyword_num_range,term_offset_range)
    # classification(past_n,start_date,end_date,keyword_num,term_offset,cv)
    all_results = np.load("all_results.npy")
    keyword_num = 50
    term_offset = 10
    x_range=[]
    y_range=[]
    for result in all_results:
        params = result['params']
        avg_score = np.mean(result['scores'])
        if(params['keyword_num']==keyword_num and params['term_offset']==term_offset):
            x_range.append(params['past_n'])
            y_range.append(avg_score)
    plot_for_varying_x("past_n",x_range,y_range)


init()
main()
# collect_news()



# todo:
# build progression into classification or vice versa
# params tuning
# read more relevant papers
# train with other models
# dedensify dates
