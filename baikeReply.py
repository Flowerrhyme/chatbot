import requests
from bs4 import BeautifulSoup


# 下载网页内容
def download(url):
    if url is None:
        return None
    # 浏览器请求头
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.96 Safari/537.36'
    headers = {'User-Agent': user_agent}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        r.encoding = 'utf-8'
        return r.text
    return None


# 提取百科词条简介
def get_data(html, keyword):
    soup = BeautifulSoup(html, 'html.parser')
    data = soup.find_all('dt', class_='basicInfo-item name')
    # meta property="og:description"
    if data != []:
        data = soup.find_all('meta', property="og:description")
        return data[0]['content']
    else:
        print("重定向，默认选第一个词条")
        soup = BeautifulSoup(html, 'html.parser')
        redirect = soup.find_all('ul', class_='custom_dot para-list list-paddingleft-1')
        redirect = redirect[0].find('a')
        redirect = redirect['data-lemmaid']
        print(redirect)
        newUrl = 'http://baike.baidu.com/item/{}/{}'.format(keyword, redirect)
        return get_data(download(newUrl), keyword)


def baike_reply(keyword):
    url = 'http://baike.baidu.com/item/{}'.format(keyword)
    html_cont = download(url)
    data = get_data(html_cont, keyword)
    return data

