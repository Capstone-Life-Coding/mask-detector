from selenium import webdriver # webdriver를 사용해서 웹 페이지의 제어를 가져온다.
from bs4 import BeautifulSoup # 웹페이지의 제어를 가져온후 BeautifulSoup로 크롤링을 진행한다.
import requests
import urllib.request
import re
import os
import time
import threading
import keyboard
import datetime


def cov_weather():
    t1 = threading.Timer(10, cov_weather)

    options = webdriver.ChromeOptions()
    options.add_argument('headless')
    options.add_argument('window-size=1920x1080')
    options.add_argument("disable-gpu")
    drivier = webdriver.Chrome('chromedriver', chrome_options=options)

    covid19 = 'http://ncov.mohw.go.kr/'
    weather = 'https://n.weather.naver.com/'

    req = urllib.request.urlopen(covid19)
    res = req.read()
    soup = BeautifulSoup(res, "html.parser")
    before = soup.find(class_='before').text

    req = urllib.request.urlopen(weather)
    res = req.read()
    soup = BeautifulSoup(res, "html.parser")
    
    todayweather = soup.find(class_='weather').text # 현재 오후 온도라 사용 x
    summary = soup.find(class_='summary').text
    todaytemp = soup.find(class_='current').text # ex) 현재 온도13
   
    #felling = soup.find(class_='desc_feeling').text # ex) 체감온도13


    before = re.findall('\d', before)
    before = "".join(before)
    result=("일일확진자 " + before + "명" + "," + todaytemp + ","+ todayweather +"," + summary.strip())

    t1.start()
    t1.cancel()
    return result


cov_weather()
if __name__ == '__main__':
    cov_weather()