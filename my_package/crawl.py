from selenium import webdriver
from bs4 import BeautifulSoup
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

    cocid19 = 'http://ncov.mohw.go.kr/'
    weather = 'https://n.weather.naver.com/'

    req = urllib.request.urlopen(cocid19)
    res = req.read()
    soup = BeautifulSoup(res, "html.parser")
    before = soup.find(class_='before').text

    req = urllib.request.urlopen(weather)
    res = req.read()
    soup = BeautifulSoup(res, "html.parser")
    todaytemp = soup.find(class_='current').text
    todayweather = soup.find(class_='weather').text

    # todayhigh=soup.find(class_='degree_height').text
    # todaylow=soup.find(class_='degree_low').text

    before = re.findall('\d', before)
    before = "".join(before)
    a=("일일확진자 " + before + '명 /' + todaytemp + "/ 날씨" + todayweather + "\n")

    t1.start()
    t1.cancel()
    return a

cov_weather()

'''
while True:
    cov_weather()
    if keyboard.is_pressed('a'): # 스페이스 키를 눌렀으면
        break
    else:
        continue'''