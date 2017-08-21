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
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics


data_root_dir = "/Users/liyuanqi/Google_Drive/UCLA_MSCS/Capstone/data"
stock_symbol = "tsla"    
training_start_date = date(2016,4,1)
training_end_date = date(2017,4,1)

def file_exists(file_path):
    return os.path.isfile(file_path) 


def get_date_duration_string(start_date,end_date):
    return start_date.strftime('%Y-%m-%d') + "_" + end_date.strftime('%Y-%m-%d')


class CustomTokenizer(object):
    def __init__(self):
        self.regexp=RegexpTokenizer(r'\b([a-zA-Z]+)\b')
        self.stemmer = LancasterStemmer()
    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in self.regexp.tokenize(doc)]


def prepare_data(past_n,start_date,end_date,keyword_num,term_offset):
    print("[data prep] preparing training data...")
    trends_data_file_name = '{0}_all_trends_{1}_offset{2}.npy'.format(data_root_dir + "/" + stock_symbol,get_date_duration_string(start_date,end_date),term_offset) 
    vectorized_data_file_name = "{0}_n{1}_vectorized_data_{2}.npy".format(data_root_dir + "/" + stock_symbol,past_n,get_date_duration_string(start_date,end_date))
    past_n_trends_matrix_file_name = "{0}_past_{1}_trends_matrix_{2}.npy".format(data_root_dir + "/" + stock_symbol,past_n,get_date_duration_string(start_date,end_date))
    
    all_news = []
    all_trends = []
    past_n_trends_matrix = []
    delta = end_date - start_date
    for i in range(delta.days + 1):
        current_date = start_date + timedelta(days=i)
        if(pc.has_date(current_date)):
            news = nc.get_news_from_past_n_days(current_date,past_n)
            news = " ".join(news[0])
            trend = pc.get_trend_by_date(current_date + timedelta(days=term_offset))
            all_news.append(news)
            all_trends.append(trend)
            trends = pc.get_past_n_trends(current_date,past_n)
            past_n_trends_matrix.append(trends)

    if(not file_exists(vectorized_data_file_name)):
        vectorizer = TfidfVectorizer(min_df=1, tokenizer=CustomTokenizer(), stop_words='english')
        print("[data prep] Vectorizing data for past_n={0}...".format(past_n))
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



def prepare_data_for_prediction(vectorizer,past_n,start_date,end_date,keyword_num,term_offset):
    print("[data prep] preparing training data...")
    trends_data_file_name = '{0}_all_trends_{1}_offset{2}.npy'.format(data_root_dir + "/" + stock_symbol,get_date_duration_string(start_date,end_date),term_offset) 
    vectorized_data_file_name = "{0}_n{1}_vectorized_data_{2}.npy".format(data_root_dir + "/" + stock_symbol,past_n,get_date_duration_string(start_date,end_date))
    past_n_trends_matrix_file_name = "{0}_past_{1}_trends_matrix_{2}.npy".format(data_root_dir + "/" + stock_symbol,past_n,get_date_duration_string(start_date,end_date))
    
    all_news = []
    all_trends = []
    past_n_trends_matrix = []
    delta = end_date - start_date
    for i in range(delta.days + 1):
        current_date = start_date + timedelta(days=i)
        if(pc.has_date(current_date)):
            news = nc.get_news_from_past_n_days(current_date,past_n)
            news = " ".join(news[0])
            trend = pc.get_trend_by_date(current_date + timedelta(days=term_offset))
            all_news.append(news)
            all_trends.append(trend)
            trends = pc.get_past_n_trends(current_date,past_n)
            past_n_trends_matrix.append(trends)

    if(not file_exists(vectorized_data_file_name)):
        print("[prediction data prep] Vectorizing data for past_n={0}...".format(past_n))
        data_train_vectorized = vectorizer.transform(all_news)
        data_train_vectorized = data_train_vectorized.toarray()
        np.save(vectorized_data_file_name,data_train_vectorized)

    np.save(trends_data_file_name,all_trends)
    np.save(past_n_trends_matrix_file_name,past_n_trends_matrix)


def classification(classifier,past_n,start_date,end_date,keyword_num,term_offset,cv):
    print("[classification] loading training data from .npy files...")
    trends_data_file_name = '{0}_all_trends_{1}_offset{2}.npy'.format(data_root_dir + "/" + stock_symbol,get_date_duration_string(start_date,end_date),term_offset) 
    vectorized_data_file_name = "{0}_n{1}_vectorized_data_{2}.npy".format(data_root_dir + "/" + stock_symbol,past_n,get_date_duration_string(start_date,end_date))
    past_n_trends_matrix_file_name = "{0}_past_{1}_trends_matrix_{2}.npy".format(data_root_dir + "/" + stock_symbol,past_n,get_date_duration_string(start_date,end_date))
    
    all_data_files_exist = file_exists(trends_data_file_name) and file_exists(vectorized_data_file_name) and file_exists(past_n_trends_matrix_file_name)
    if(not all_data_files_exist):
        prepare_data(past_n,start_date,end_date,keyword_num,term_offset)

    data_train_vectorized = np.load(vectorized_data_file_name)
    all_trends = np.load(trends_data_file_name)
    past_n_trends_matrix = np.load(past_n_trends_matrix_file_name)

    # print("Trend Ratio: {0} Increases (True) : {1} Decreases (False)").format(np.sum(all_trends), len(all_trends) - np.sum(all_trends))
    lsi = TruncatedSVD(n_components=keyword_num, random_state=42)
    data_train_matrix = lsi.fit_transform(data_train_vectorized)
    data_train_matrix_with_past_price = np.c_[data_train_matrix, past_n_trends_matrix]
    # classifier = LogisticRegression()
    print("[classification] training and predicting with past_n={0},keyword_num={1},term_offset={2}...".format(past_n,keyword_num,term_offset))
    scores = cross_val_score(classifier, data_train_matrix_with_past_price, all_trends, cv=cv,scoring='average_precision')
    return data_train_matrix_with_past_price, all_trends, scores


def get_vectorizer_and_lsi_for_period(start_date,end_date,past_n,keyword_num):
    all_news = []
    delta = end_date - start_date
    for i in range(delta.days + 1):
        current_date = start_date + timedelta(days=i)
        if(pc.has_date(current_date)):
            news = nc.get_news_from_past_n_days(current_date,past_n)
            news = " ".join(news[0])
            all_news.append(news)
    vectorizer = TfidfVectorizer(min_df=1, tokenizer=CustomTokenizer(), stop_words='english')
    data_train_vectorized = vectorizer.fit_transform(all_news)

    lsi = TruncatedSVD(n_components=keyword_num, random_state=42)
    data_train_matrix = lsi.fit_transform(data_train_vectorized)
    return vectorizer,lsi


def predict_with_classifier(classifier,past_n,start_date,end_date,keyword_num,term_offset):
    print("[prediction] loading training data from .npy files...")
    trends_data_file_name = '{0}_all_trends_{1}_offset{2}.npy'.format(data_root_dir + "/" + stock_symbol,get_date_duration_string(start_date,end_date),term_offset) 
    vectorized_data_file_name = "{0}_n{1}_vectorized_data_{2}.npy".format(data_root_dir + "/" + stock_symbol,past_n,get_date_duration_string(start_date,end_date))
    past_n_trends_matrix_file_name = "{0}_past_{1}_trends_matrix_{2}.npy".format(data_root_dir + "/" + stock_symbol,past_n,get_date_duration_string(start_date,end_date))
    lsi = None
    all_data_files_exist = file_exists(trends_data_file_name) and file_exists(vectorized_data_file_name) and file_exists(past_n_trends_matrix_file_name)
    if(not all_data_files_exist):
        vectorizer,lsi = get_vectorizer_and_lsi_for_period(training_start_date,training_end_date,past_n,keyword_num)
        prepare_data_for_prediction(vectorizer,past_n,start_date,end_date,keyword_num,term_offset)

    data_train_vectorized = np.load(vectorized_data_file_name)
    all_trends = np.load(trends_data_file_name)
    past_n_trends_matrix = np.load(past_n_trends_matrix_file_name)
    data_train_matrix = lsi.transform(data_train_vectorized)
    data_train_matrix_with_past_price = np.c_[data_train_matrix, past_n_trends_matrix]
    print(data_train_matrix_with_past_price.shape)
    print("[prediction] predicting with past_n={0},keyword_num={1},term_offset={2}...".format(past_n,keyword_num,term_offset))
    predicted_labels = classifier.predict(data_train_matrix_with_past_price)
    print(metrics.classification_report(all_trends, predicted_labels))


def get_cross_val_score(classifier,data_train,labels,cv):
    try:
        scores = cross_val_score(classifier, data_train, labels, cv=cv,scoring='average_precision')
    except Exception:
        param = {'silent':1, 'objective':'binary:logistic'}
        num_round=1
        res = classifier.cv(param, dtrain, num_round, nfold=5,
               metrics={'error'}, seed = 0)
        print(res)

def train_with_params(classifier,start_date,end_date,cv,past_n_range,keyword_num_range,term_offset_range):
    all_results = []
    for past_n in past_n_range:
        for keyword_num in keyword_num_range:
            for term_offset in term_offset_range:
                params = {}
                params['past_n'] = past_n
                params['keyword_num'] = keyword_num
                params['term_offset'] = term_offset
                training_data, training_labels, scores = classification(classifier,past_n,start_date,end_date,keyword_num,term_offset,cv)
                classifier.fit(training_data,training_labels)

                print(training_data.shape)

                prediction_start_date = date(2017,4,2)
                prediction_end_date = date(2017,5,1)

                predict_with_classifier(classifier,past_n,prediction_start_date,prediction_end_date,keyword_num,term_offset)
                result = {}
                result['params'] = params
                result['scores'] = scores
                all_results.append(result)
                print_and_write_to_file("** | past_n={0} | keyword_num={1} | term_offset={2} | AVG_SCORE = {3}"
                    .format(result['params']['past_n'],result['params']['keyword_num'],result['params']['term_offset'],np.mean(result['scores'])))
                print_and_write_to_file("** scores: [{0}]".format(', '.join([str(x) for x in result['scores']])))
                print_and_write_to_file("-------------------------------------------------------------------------")
                
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


def init(data_root_dir,stock_symbol):
    nc.init(data_root_dir,stock_symbol)
    pc.init()
    try:
        os.remove('all_results.txt')
    except OSError:
        pass


def main():
    init(data_root_dir,stock_symbol)

    cv = 5
    # real dates
    start_date = training_start_date
    end_date = training_end_date
    # real ranges
    # past_n_range = np.arange(1,41,1)
    # keyword_num_range = np.arange(50,160,30)
    # term_offset_range = [1,5,10,15,20,25,30]
    past_n_range = [6]
    keyword_num_range = [100]
    term_offset_range = [10]
    # classifier = RandomForestClassifier(random_state=1)
    # classifier = XGBClassifier(seed=1)
    classifier = MLPClassifier(solver='lbfgs',random_state=1)
    train_with_params(classifier,start_date,end_date,cv,past_n_range,keyword_num_range,term_offset_range)
    

    # all_results = np.load("all_results.npy")
    # keyword_num = 100
    # term_offset = 10
    # x_range=[]
    # y_range=[]
    # for result in all_results:
    #     params = result['params']
    #     avg_score = np.mean(result['scores'])
    #     if(params['keyword_num']==keyword_num and params['term_offset']==term_offset):
    #         x_range.append(params['past_n'])
    #         y_range.append(avg_score)
    # plot_for_varying_x("past_n",x_range,y_range)


main()
