from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
# from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import ElementNotVisibleException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.common.exceptions import NoSuchElementException
import urllib.request
import os
from os import path
from main import read_and_image

def save_images(car):
    print("here")
    image_list = []
    image_names_list = []
    driver.get(car)
    date = driver.find_element_by_class_name('date')
    date = date.text.split('.')
    date = date[2][0:4] + '.' + date[1]
    print(date)
    for img in driver.find_elements_by_xpath('//a[contains(@href, "/img/")]'):
        image_list.append(img.get_attribute("href"))
        # print(img.get_attribute("href"))
        newimg_name = img.get_attribute("href").split('.')
        newimg_name = newimg_name[2].split("=")
        newimg_name = newimg_name[1]
        image_names_list.append(newimg_name)
    image_list = list(dict.fromkeys(image_list))
    image_names_list = list(dict.fromkeys(image_names_list))
    print(image_list)
    print(image_names_list)
    adname = image_names_list[0].split('_')[0]
    with open('checked.txt') as f:
        if adname in f.read():
            print("true")
            f.close()
        else:
            f.close()
            if path.exists("./" + date) is False:
                os.mkdir("./" + date)
            for image in image_list:
                urllib.request.urlretrieve(image, "./" + date + "/" + image_names_list[image_list.index(image)] + '.jpg')
                read_and_image("./" + date + "/" + image_names_list[image_list.index(image)] + '.jpg',  date + '/' )
            file = open('checked.txt', "a")
            file.write(adname + '\n')
            file.close()





def find_car(marks_link):
    car_list = []
    true_car_list = []
    driver.get(marks_link)
    # driver.find_element_by_xpath('//a[@href="' + marks_link[18:] + '"]').click()
    # print(marks_link[23:])
    for ad in driver.find_elements_by_xpath('//a[contains(@href,"'+ marks_link[23:] +'")]'):
        car_list.append(ad.get_attribute("href"))
    # print(car_list)
    for car in car_list:
        # print(car[18:30])
        # print(marks_link[18:30])
        # print(car[-6:-1])
        if car[-5:-1] == '.htm':
            true_car_list.append(car)
    true_car_list = list(dict.fromkeys(true_car_list))
    print(true_car_list)
    for true_car in true_car_list:
        print(true_car[18:])
        # driver.find_element_by_xpath('//a[@href="' + true_car[18:] + '"]').click()
        save_images(true_car)
        # # driver.wait = WebDriverWait(driver, 5)
        # for img in driver.find_elements_by_xpath('//a[contains(@href, "/img/")]'):
        #     print(img.get_attribute("href"))
        # driver.execute_script("window.history.go(-1)")
        # driver.execute_script("window.history.go(-1)")
        driver.get(marks_link)
        print(driver.current_url)
    # driver.execute_script("window.history.go(-2)")



# driver = webdriver.Chrome('./chromedriver')  # если запускать с локальной машины
# options.add_argument('headless')  # для открытия headless-браузера
# browser = webdriver.Chrome(executable_path=chromedriver, chrome_options=options)chrome_options = webdriver.ChromeOptions()
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--ignore-certificate-errors')
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument("--remote-debugging-port=9222")  # this
driver = webdriver.Chrome(executable_path='./chromedriver',options=chrome_options) #если запускать с локальной машины
# driver = webdriver.Chrome(ChromeDriverManager().install())
driver.wait = WebDriverWait(driver, 5)
driver.get('https://forsage.by/')
marks_links = []
# Поиск тегов по имени
# links = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.LINK_TEXT, "/auto/acura/")))
# references = [link.get_attribute("href") for link in links]
# print(references)
for banners in  driver.find_elements_by_class_name('list-auto-without-logo'):
    # marks_list = driver.find_element_by_class_name('wrapper clearfix')
    elems = driver.find_elements_by_xpath("//a[contains(@href, '/auto/')]")
    # elems = driver.findElement(By.cssSelector("a[href*='/auto/']"))
    for elem in elems:
        marks_links.append(elem.get_attribute("href"))
    print(marks_links)
    begin = marks_links.index('https://forsage.by/auto/acura/')
    end = marks_links.index('https://forsage.by/auto/uaz/')
    marks_links = marks_links[begin:end]
    print(marks_links)
    for marks_link in marks_links:
        find_car(marks_link)
        # driver.find_element_by_xpath('//a[@href="' + marks_link[18:] + '"]').click()
        # # elems = driver.find_elements_by_xpath("//a[contains(@href, '{marks_link[23:]}')]")
        # for ad in driver.find_elements_by_xpath("//a[contains(@href, '{marks_link[23:]}')]"):
        #     print(ad.get_attribute("href"))
        # for ad in driver.find_element_by_xpath('//a[@href="' + marks_link[23:] + '"]'):
        #     print(ad.get_attribute("href"))



# password = browser.find_element_by_name('ctl00$MainContent$ctlLogin$_Password')
# login = browser.find_element_by_name('ctl00$MainContent$ctlLogin$BtnSubmit')