import bs4
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options

def get_page_content(url):
    page = requests.get(url)
    return bs4.BeautifulSoup(page.text, "html.parser")

def get_links(data):
    files = data.find_all("a")
    return [link.attrs["href"] for link in files]

def links_standardlize(links, url):
    for i in range(len(links)):
        if links[i][0] == '/':
            links[i] = url + links[i] 

    return links

def get_csv_from_pdf(driver, url, link):
    driver.get(url)
    driver.find_element_by_id("inputFile").send_keys(link)
    driver.find_element_by_id("toExtensionSel").send_keys("csv")
    driver.find_element_by_id("convertButton").click()
    alert = driver.switch_to.alert
    alert.accept()

    time.sleep(20)
    driver.find_element_by_link_text("Download").click()

    print("Write content to csv file successfully: " + link)

def get_txt_from_url(url, index):
    page = get_page_content(url)
    content = page.find("div", class_="content-main").get_text()

    write_content_to_file(content, index)

    print("Write content to txt file successfully:" + url)

def write_content_to_file(content, index):
    f = open("data" + str(index), "w")
    f.write(content)
    f.close()

def crawl(links):
    CHROME_PATH = '/usr/bin/google-chrome'
    CHROMEDRIVER_PATH = '/usr/bin/chromedriver'
    WINDOW_SIZE = "1920,1080"

    chrome_options = Options()  
    chrome_options.add_argument("--headless")  
    chrome_options.add_argument("--window-size=%s" % WINDOW_SIZE)
    chrome_options.binary_location = CHROME_PATH

    driver = webdriver.Chrome(executable_path=CHROMEDRIVER_PATH,
                              options=chrome_options)  

    for link in links:
        if link[-4:] == ".pdf":
            try:
                get_csv_from_pdf(driver, "https://www.zamzar.com/url/", link)
            except:
                print("Crawl failed, link: " + link)
        else:
            try:
                get_txt_from_url(link, links.index(link))
            except:
                print("Crawl failed, link: " + link)
        

def main():
    url = "https://tdtu.edu.vn/giao-duc/cong-khai-thong-tin"

    soup = get_page_content(url)

    data = soup.find("div", class_="node__content")

    links = get_links(data)

    links_standardlized = links_standardlize(links, "https://tdtu.edu.vn")

    crawl(links_standardlized)

if __name__ == "__main__":
    main()