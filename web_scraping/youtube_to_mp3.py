import time, datetime
from selenium import webdriver
from bs4 import BeautifulSoup as bs


# Change before use
browser = webdriver.Chrome('/Users/hongtan/Downloads/chromedriver')

# search = "angry joe"

# browser.get(f'https://www.youtube.com/results?search_query={search}')

browser.get('https://en.y2mate.is/95/')

time.sleep(2)

youtube_link = 'https://www.youtube.com/watch?v=qZ90POH8JlA'

# Type link into input field
link = browser.find_element_by_id('txtUrl')
link.send_keys(youtube_link)

# Start the conversion to mp3
start = browser.find_element_by_xpath('/html/body/section[1]/div/div[2]/div[1]/div/form/button/span[1]')
start.click()

time.sleep(2)

# Go into audio (mp3) section
audio = browser.find_element_by_xpath('/html/body/section[1]/div/div[2]/div[2]/div/div[2]/div/div[1]/button[2]')
audio.click()

# Click the convert button for 160k quality
btn = browser.find_element_by_xpath('/html/body/section[1]/div/div[2]/div[2]/div/div[2]/div/div[3]/table/tbody/tr[2]/td[3]/button')
btn.click()

time.sleep(1)

# Click the download button
downloadbtn = browser.find_element_by_xpath('/html/body/section[1]/div/div[2]/div[2]/div/div[2]/div/div[3]/table/tbody/tr[2]/td[3]/button/a')
downloadbtn.click()

# link.click()