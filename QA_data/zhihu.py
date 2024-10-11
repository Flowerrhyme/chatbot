#-*- coding: UTF-8 -*-
# 爬取文章
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import jieba

#获取网页body里的内容
def get_content(url , data = None):
    # 设置Http请求头，根据自己电脑查一下
    header={
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Encoding': 'gzip, deflate, sdch',
        'Accept-Language': 'zh-CN,zh;q=0.8',
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.235'
    }

    req = requests.get(url, headers=header)
    req.encoding = 'utf-8'
    bs = BeautifulSoup(req.text, "html.parser")  # 创建BeautifulSoup对象
    body = bs.body #

    return body

#获取问题标题
def get_title(html_text):
    data = html_text.find('h1', {'class':'QuestionHeader-title'})  #匹配标签
    return data.string.encode('utf-8')

#获取问题内容
def get_question_content(html_text):
    data = html_text.find('span', {'class': 'RichText ztext'})
    print (data.string)
    if data.string is None:
        out = ''
        for datastring in data.strings:
            datastring = datastring.encode('utf-8')
            out = out + datastring.encode('utf-8')
        print ('内容：\n' + out)
    else:
        print ('内容：\n' + data.string.encode('utf-8'))

#获取点赞数
def get_answer_agree(body):
    agree = body.find('button',{'class': 'Button VoteButton VoteButton--up'})
    agree_html = BeautifulSoup(str(agree), "html.parser")
    all_buttons = agree_html.find_all("button", {"class": "Button VoteButton VoteButton--up"})
    one_button = all_buttons[0]
    agree_number = one_button["aria-label"]
    print(agree_number)

#获取答案
def get_response(html_text):
    out1 = ''
    response = html_text.find_all('div', {'class': 'ContentItem-time'})
    for index in range(len(response)):
        #获取标签
        answerhref = response[index].find('a', {'target': '_blank'})
        if not(answerhref['href'].startswith('javascript')):
            url = 'http:' + answerhref['href']
            body = get_content(url)
            get_answer_agree(body)
            answer = body.find('span', {'class': 'RichText ztext CopyrightRichText-richText css-ql0744'})
            
            if answer.string is None:
                out = ''
                for datastring in answer.strings:
                    datastring = datastring.encode('utf-8')
                    out = out + '\n' + str(datastring,encoding = 'utf-8')
            else:
                print (answer.string.encode('utf-8'))
        out1 =  out
    return  out1


def get_question(filename):
    URL_target=[]
    f = open(filename,encoding = 'utf-8')# 返回一个文件对象   
    line = f.readline()
    for line in f:
        URL_target.append(line[:-1])#去掉换行符
    f.close()
    return URL_target


# #   输入要爬取的网址
# URL_target = 'https://www.zhihu.com/question/505503990/answer/2276487889'
# html_text = get_content(URL_target)
# title = get_title(html_text)
# print ("标题：" + str(title,encoding = 'utf-8') + '\n')
# title_utf8=str(title,encoding = 'utf-8')
# data = get_response(html_text)
# print (data)

# df = pd.DataFrame(columns=['ID','Q','A'])#新建一个dataframe

# URL_target=['https://www.zhihu.com/question/505503990/answer/2276487889','https://www.zhihu.com/question/453282676/answer/2420180142'
# ,'https://www.zhihu.com/question/498993604/answer/2255720199']

# id=1
# result_list=[]

# for url in URL_target:
#     html_text = get_content(url)
#     title = get_title(html_text)
#     print ("标题：" + str(title,encoding = 'utf-8') + '\n')
#     title_utf8=str(title,encoding = 'utf-8')
#     data = get_response(html_text)
#     #print (data)

#     result_list.append([id,title_utf8, data])#保存到结果表中
#     id=id+1

# for result in result_list:
#     df = df.append(pd.DataFrame({"ID":[result[0]], "Q":[result[1]], "A":[result[2]]}), ignore_index=True)
# df.to_csv('new_QA2.csv')

# data2=pd.read_csv('new_QA2.csv')
# data2 = data2["A"]
# data2=data2.tolist()
# for i in data2:
#     print(i)

if __name__ == "__main__":
    stop_words = []
    with open('./QA_data/stop_words.txt', encoding='gbk') as f:
        for line in f.readlines():
            stop_words.append(line.strip('\n'))

    df = pd.DataFrame(columns=['ID','Q','A','TAG'])#新建一个dataframe

    URL_target=get_question('QA_data/URL_target_list.txt')
    print(URL_target)

    id=107
    result_list=[]

    for url in URL_target:
        html_text = get_content(url)
        title = get_title(html_text)
        print ("标题：" + str(title,encoding = 'utf-8') + '\n')
        title_utf8=str(title,encoding = 'utf-8')
        data = get_response(html_text)
        #print (data)

        #得到tag
        question = list(jieba.cut(title_utf8, cut_all=False)) #对查询字符串进行分词
        for word in reversed(question):  #去除停用词
            if word in stop_words:
                question.remove(word)
        newtag=''
        for tag in question: #按照每个tag，循环构造查询语句
            newtag=newtag+'|'+tag
        newtag=newtag[1:]

        result_list.append([id,title_utf8, data,newtag])#保存到结果表中
        id=id+1

    for result in result_list:
        df = df.append(pd.DataFrame({"ID":[result[0]], "Q":[result[1]], "A":[result[2]],"TAG":[result[3]]}), ignore_index=True)
    df.to_csv('new_QA.csv',index = False)





