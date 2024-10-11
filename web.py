# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import time
import jieba
import os
from datapreprocess import preprocess
import train_eval
from QA_data import QA_test
from config import Config
from aip import AipSpeech
import json
from random import choice
from baikeReply import baike_reply
import shutil

history = []


# 语音合成
def voice_output(res_msg, cur_time):
    # 设置登录信息
    APP_ID = '25865801'
    API_KEY = '2GaqYNAHnRa7V8D6Iv12w6E5'
    SECRET_KEY = 'WKHCW8o6g8TX5Cbi5KhuzA0Xh337XNzG'
    client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

    # print(res_msg)

    # 合成语音
    result = client.synthesis(res_msg, 'zh', 1, {
        'vol': 5,  # 音量
        'pit': 5,  # 语调
        'per': 0,  # 发音人
        'spd': 5,  # 语速
    })

    # 保存返回的音频流，默认MP3格式
    path = './static/res/cache/tempOutput{}.mp3'.format(cur_time)
    if not isinstance(result, dict):
        with open(path, 'wb') as f:  # 创建mp3文件并具有写权限，用二进制的方式打开
            f.write(result)
    # result = str(result)
    # print(type(result), result)
    return path


# 使用beamSearch解码预测
def predict(req_msg, **kwargs):
    opt = Config()
    for k, v in kwargs.items():  # 设置参数
        setattr(opt, k, v)
    searcher, sos, eos, unknown, word2ix, ix2word = train_eval.test(opt)

    if os.path.isfile(opt.corpus_data_path) == False:
        preprocess()

    return train_eval.output_answer(req_msg, searcher, sos, eos, unknown, opt, word2ix, ix2word)


# 使用nucleus filtering解码预测
def predict2(req_msg, **kwargs):
    global history
    this_history = history
    opt = Config()
    for k, v in kwargs.items():  # 设置参数
        setattr(opt, k, v)
    encoder, decoder, sos, eos, unknown, word2ix, ix2word = train_eval.test2(opt)

    if os.path.isfile(opt.corpus_data_path) == False:
        preprocess()

    output_words, history = train_eval.filter_answer(req_msg, encoder, decoder, sos, eos, unknown, opt, word2ix,
                                                     ix2word,
                                                     this_history)
    return output_words


app = Flask(__name__, static_url_path="/static")

# 删除暂存的音频文件
shutil.rmtree('static/res/cache')
os.mkdir('static/res/cache')


@app.route('/message', methods=['POST'])
# """定义应答函数，用于获取输入信息并返回相应的答案"""
def reply(**kwargs):
    # 从请求中获取参数信息
    req_msg = request.form['msg']
    req_copy = req_msg
    print(req_msg)

    # 将语句使用结巴分词进行分词
    req_msg = " ".join(jieba.cut(req_msg))

    '''
    生成回答信息，分下列情况：
    1. 以#号开头强制不适用数据库知识；
    2. 以@号开头使用百度百科知识回答；
    3. 正常情况默认优先匹配数据库答案，没有匹配则用神经网络生成
    '''
    if req_msg[0] == '#':
        # print("强制神经网络回答")
        res_msg = predict(req_msg)
    elif req_msg[0] == '@':
        res_msg = baike_reply(req_copy[1:])
        # print('百科查询结果', req_msg[1], res_msg)
    else:
        # print("开始查询数据库")
        query_res = QA_test.match(req_msg)
        if query_res == tuple():
            # print("无匹配，使用神经网络生成")
            res_msg = predict(req_msg)
            # res_msg = predict2(req_msg)
        else:
            # print("有匹配，选择回答")
            answers = query_res[2].split(sep='|')
            res_msg = choice(answers)

    # 将unk值的词用微笑符号代替
    res_msg = res_msg.replace('_UNK', '^_^')
    res_msg = res_msg.strip()

    # 语音合成
    voice_path = voice_output(res_msg, time.time())

    return jsonify({'text': res_msg, 'voice': voice_path})


@app.route("/")
def index():
    return render_template("index.html")  # 返回渲染的网页


# 启动
if (__name__ == "__main__"):
    app.run(host='127.0.0.1', port=5000)
