import twitter
import urllib2

import random
import time
import re
import gzip, StringIO
import os.path
from datetime import date, timedelta
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from nytimesarticle import articleAPI
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


user_agents = list()


def get_ariticle_urls_for_date(**params):
    URL_template = "https://www.google.com/search?q={stock_symbol}+stock&tbs=cdr%3A1%2Ccd_min%3A{month}%2F{day}%2F{year}%2Ccd_max%3A{month}%2F{day}%2F{year}&as_mindate={month}%2F%{day}%2F{year}&as_maxdate={month}%2F{day}%2F{year}&tbm=nws"
    URL = URL_template.format(**params)
    driver = webdriver.Firefox()
    driver.get(URL)
    links = []
    elems = driver.find_elements_by_class_name("_PMs")
    for x in range(0,len(elems)):
        links.append(elems[x].get_attribute("href"))
    driver.close()
    return links


def get_html_with_url(url):
    retry = 3
    html = ""
    while(retry > 0):
        try:
            request = urllib2.Request(url)
            length = len(user_agents)
            index = random.randint(0, length-1)
            user_agent = user_agents[index] 
            request.add_header('User-agent', user_agent)
            request.add_header('connection','keep-alive')
            request.add_header('Accept-Encoding', 'gzip')
            request.add_header('referer', "https://www.google.com/")
            response = urllib2.urlopen(request)
            html = response.read() 
            if(response.headers.get('content-encoding', None) == 'gzip'):
                html = gzip.GzipFile(fileobj=StringIO.StringIO(html)).read()
            break;
        except urllib2.URLError,e:
            print 'url error:', e
            random_sleep(5,10)
            retry = retry - 1
            continue
        
        except Exception, e:
            print 'error:', e
            retry = retry - 1
            random_sleep(5,10)
            continue
    return html


def random_sleep(lower,upper):
    sleeptime =  random.randint(lower, upper)
    print("[INFO] Sleeping for {0} seconds...".format(sleeptime))
    time.sleep(sleeptime)


def load_user_agent():
    fp = open('./user_agents', 'r')

    line  = fp.readline().strip('\n')
    while(line):
        user_agents.append(line)
        line = fp.readline().strip('\n')
    fp.close()


def visible(element):
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('<!--.*-->', str(element.encode('utf-8'))):
        return False
    return True


def get_text_from_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    data = soup.findAll(text=True)
    result = filter(visible, data)
    return result
    # return soup.get_text()


def save_html_to_file(root_dir,stock_symbol,html,year,month,day,index):
    with open("{0}/{1}_{2}-{3}-{4}_news{5}.html".format(root_dir,stock_symbol,year,month,day,index), "w") as text_file:
        text_file.write(html)


def collect_news(root_dir,stock_symbol,start_date,end_date):
    nc.init()

    delta = end_date - start_date         # timedelta

    for i in range(delta.days + 1):
        current_date = start_date + timedelta(days=i)
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


def get_news_from_past_n_days(root_dir,stock_symbol,date,n):
    news = []
    current_date = date
    for i in range(1,n+1):
        past_date = current_date-timedelta(days=i)
        print("now reading news for date: {0}-{1}-{2}".format(past_date.year,past_date.month,past_date.day))

        prefix = "{0}/{1}_{2}-{3}-{4}_news".format(root_dir,stock_symbol,past_date.year,past_date.month,past_date.day)
        for j in range(0,10):
            news_html_file = "{0}{1}{2}".format(prefix,j,".html")
            if(os.path.isfile(news_html_file)):
                with open(news_html_file) as f:
                    # print("{0}{1}".format("now reading from: ",news_html_file))
                    text = get_text_from_html(f)
                    text = ' '.join(text)
                    news.append(text)
    return news


def init():
    load_user_agent()

# root_dir = "/Users/liyuanqi/Google_Drive/UCLA_MSCS/Capstone/data"
# stock_symbol = "tsla"
# get_news_from_past_n_days(root_dir,stock_symbol,2016,7,1,1)
