import twitter
import urllib2

import random
import time
import re
import gzip, StringIO

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from nytimesarticle import articleAPI
from bs4 import BeautifulSoup


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
    soup = BeautifulSoup(html)
    data = soup.findAll(text=True)
    result = filter(visible, data)
    # print(result)
    # print list(result)


def save_html_to_file(root_dir,stock_symbol,html,year,month,day,index):
    with open("{0}/{1}_{2}-{3}-{4}_news{5}.html".format(root_dir,stock_symbol,year,month,day,index), "w") as text_file:
        text_file.write(html)


def init():
    load_user_agent()
