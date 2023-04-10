from bs4 import BeautifulSoup as bs
import requests
from selenium import webdriver
import time
import csv
import numpy as np
import os

#Change the working directory 
path = input("Please enter the path: ")
os.chdir(path)
#Take user input to search for audio file
search = input("Search for audio files related to: ").replace(" ", "+")

driver =webdriver.Chrome()

try:
    url = "https://www.youtube.com/results?search_query=" + search
except:
    print("Invalid Search")

driver.get(url)
for i in range(10):
    time.sleep(1)
    driver.execute_script(f"window.scrollTo(0,10000*{i})")

soup = bs(driver.page_source, 'html.parser')
driver.quit()
res = soup.find_all('a', id = 'thumbnail')

for thumbnail in res:
    if(thumbnail.get("href")):
        os.system("yt-dlp -x --audio-format mp3 https://www.youtube.com" + thumbnail.get("href"))

        








