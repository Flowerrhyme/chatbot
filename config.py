# -*- coding: utf-8 -*- 

import torch


class Config:
    '''
    nucleus filtering参数
    '''
    repetition_penalty = 1.0  # 重复惩罚参数，若生成的对话重复性较高，可适当提高该参数
    max_history_len = 10
    temperature = 1.0
    # 最高k选1
    topk = 8
    # 最高积累概率
    topp = 0
    '''
    模型参数
    '''
    corpus_data_path = 'corpus.pth'  # 已处理的对话数据
    # use_QA_first = True  # 是否载入知识库
    max_input_length = 50  # 输入的最大句子长度
    max_generate_length = 20  # 生成的最大句子长度
    prefix = 'checkpoints/chatbot'  # 模型断点路径前缀
    model_ckpt = ''  # 加载模型路径
    '''
    训练参数
    '''
    batch_size = 2048
    shuffle = True  # dataloader是否打乱数据
    num_workers = 0  # dataloader多进程提取数据
    bidirectional = True  # Encoder-RNN是否双向
    hidden_size = 256
    embedding_dim = 256
    method = 'dot'  # attention method
    dropout = 0  # 是否使用dropout
    clip = 50.0  # 梯度裁剪阈值
    num_layers = 2  # Encoder-RNN层数
    learning_rate = 1e-3
    teacher_forcing_ratio = 1.0  # teacher_forcing比例
    decoder_learning_ratio = 5.0
    use_gpu = False # 是否使用gpu
    device = torch.device("cuda" if use_gpu else "cpu")  # device
    '''
    训练周期信息
    '''
    epoch = 6000
    print_every = 1  # 每隔print_every个Iteration打印一次
    save_every = 50  # 每隔save_every个Epoch打印一次

