# -*- coding: utf-8 -*- 
import re
import time
import random
import jieba
import torch
import logging
from model import EncoderRNN, LuongAttnDecoderRNN
from searcher.beamsearch import BeamSearchDecoder
from searcher.greedysearch import GreedySearchDecoder
from dataload import get_dataloader
from config import Config
import torch.nn.functional as F

jieba.setLogLevel(logging.INFO)  # 关闭jieba输出信息


# 损失函数
def maskNLLLoss(input, target, mask):
    '''
    input: shape [batch_size,voc_length]
    target: shape [batch_size] 经过view ==> [batch_size, 1] 这样就和inp维数相同，可以用gather
        target作为索引,在dim=1上索引input的值,得到的形状同target [batch_size, 1]
        然后压缩维度,得到[batch_size], 取负对数后
        选择那些值为1的计算loss, 并求平均，得到loss
        故loss实际是batch_size那列的均值，表示一个句子在某个位置(t)上的平均损失值
        故nTotal表示nTotal个句子在某个位置上有值
    mask: shape [batch_size]
    loss: 平均一个句子在t位置上的损失值
    '''
    nTotal = mask.sum()  # padding的词是0，非padding的词是1，取sum得到词的个数
    crossEntropy = -torch.log(torch.gather(input, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    return loss, nTotal.item()


# 单个batch训练过程
def train_by_batch(sos, opt, data, encoder_optimizer, decoder_optimizer, encoder, decoder):
    # 清空梯度

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # 处理一个batch数据
    inputs, targets, mask, input_lengths, max_target_length, indexes = data
    inputs = inputs.to(opt.device)
    targets = targets.to(opt.device)
    mask = mask.to(opt.device)
    input_lengths = input_lengths.to(opt.device)

    # 初始化变量
    loss = 0
    print_losses = []
    n_totals = 0

    # forward计算
    '''
    inputs: shape [max_seq_len, batch_size]
    input_lengths: shape [batch_size]
    encoder_outputs: shape [max_seq_len, batch_size, hidden_size]
    encoder_hidden: shape [num_layers*num_directions, batch_size, hidden_size]
    decoder_input: shape [1, batch_size]
    decoder_hidden: decoder的初始hidden输入,是encoder_hidden取正方向
    '''
    # 计算encoder输出
    encoder_outputs, encoder_hidden = encoder(inputs, input_lengths)

    # 计算decoder输入（全是起始符）
    decoder_input = torch.LongTensor([[sos for _ in range(opt.batch_size)]])
    decoder_input = decoder_input.to(opt.device)
    decoder_hidden = encoder_hidden[:decoder.num_layers]

    # 确定是否teacher forcing
    use_teacher_forcing = True if random.random() < opt.teacher_forcing_ratio else False
    '''
    一次处理一个时刻，即一个字符
    decoder_output: [batch_size, voc_length]
    decoder_hidden: [decoder_num_layers, batch_size, hidden_size]
    如果使用teacher_forcing,下一个时刻的输入是当前正确答案，即
    targets[t] shape: [batch_size] ==> view后 [1, batch_size] 作为decoder_input
    '''

    # 分情况计算decoder输出
    if use_teacher_forcing:
        for t in range(max_target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_input = targets[t].view(1, -1)
            # 计算累计的loss
            '''
            每次迭代,
            targets[t]: 一个batch所有样本指定位置(t位置)上的值组成的向量，shape [batch_size]
            mask[t]: 一个batch所有样本指定位置(t位置)上的值组成的向量, 值为1表示此处未padding
            decoder_output: [batch_size, voc_length]
            '''
            mask_loss, nTotal = maskNLLLoss(decoder_output, targets[t], mask[t])
            mask_loss = mask_loss.to(opt.device)
            loss += mask_loss
            '''
            这里loss在seq_len方向迭代进行累加, 最终得到一个句子在每个位置的损失均值之和
            总结:  mask_loss在batch_size方向累加,然后求均值,loss在seq_len方向进行累加
            即: 一个batch的损失函数: 先计算所有句子在每个位置的损失总和,再除batch_size
            这里的loss变量用于反向传播，得到的是一个句子的平均损失
            '''
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # 不是teacher forcing: 下一个时刻的输入是当前模型预测概率最高的值
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(opt.batch_size)]])
            decoder_input = decoder_input.to(opt.device)
            # 计算累计的loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, targets[t], mask[t])
            mask_loss = mask_loss.to(opt.device)
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # 反向传播
    loss.backward()

    # 梯度裁剪
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), opt.clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), opt.clip)

    # 更新参数
    encoder_optimizer.step()
    decoder_optimizer.step()
    # 返回batch中一个位置(视作二维矩阵一个格子)的平均损失
    return sum(print_losses) / n_totals


def train(**kwargs):
    opt = Config()
    for k, v in kwargs.items():  # 设置参数
        setattr(opt, k, v)

        # 数据
    dataloader = get_dataloader(opt)
    _data = dataloader.dataset._data
    word2ix = _data['word2ix']
    sos = word2ix.get(_data.get('sos'))
    voc_length = len(word2ix)

    # 定义模型
    encoder = EncoderRNN(opt, voc_length)
    decoder = LuongAttnDecoderRNN(opt, voc_length)

    # 加载断点,从上次结束地方开始
    if opt.model_ckpt:
        checkpoint = torch.load(opt.model_ckpt)
        encoder.load_state_dict(checkpoint['en'])
        decoder.load_state_dict(checkpoint['de'])

    # 切换模式（启用batch normalization和drop out）
    encoder = encoder.to(opt.device)
    decoder = decoder.to(opt.device)
    encoder.train()
    decoder.train()

    # 定义优化器
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=opt.learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=opt.learning_rate * opt.decoder_learning_ratio)
    if opt.model_ckpt:
        encoder_optimizer.load_state_dict(checkpoint['en_opt'])
        decoder_optimizer.load_state_dict(checkpoint['de_opt'])

        # 定义打印loss的变量
    print_loss = 0

    for epoch in range(opt.epoch):
        for ii, data in enumerate(dataloader):
            # 取一个batch训练
            loss = train_by_batch(sos, opt, data, encoder_optimizer, decoder_optimizer, encoder, decoder)
            print_loss += loss
            # 打印损失
            if ii % opt.print_every == 0:
                print_loss_avg = print_loss / opt.print_every
                print("Epoch: {}; Epoch Percent complete: {:.1f}%; Average loss: {:.4f}"
                      .format(epoch, epoch / opt.epoch * 100, print_loss_avg))
                print_loss = 0

        # 保存checkpoint
        if epoch % opt.save_every == 0:
            checkpoint_path = '{prefix}_{time}'.format(prefix=opt.prefix,
                                                       time=time.strftime('%m%d_%H%M'))
            torch.save({
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
            }, checkpoint_path)


def generate(input_seq, searcher, sos, eos, opt):
    # input_seq: 已分词且转为索引的序列
    # input_batch: shape: [1, seq_len] ==> [seq_len,1] (即batch_size=1)
    input_batch = [input_seq]
    input_lengths = torch.tensor([len(seq) for seq in input_batch])
    input_batch = torch.LongTensor([input_seq]).transpose(0, 1)
    input_batch = input_batch.to(opt.device)
    input_lengths = input_lengths.to(opt.device)
    tokens, scores = searcher(sos, eos, input_batch, input_lengths, opt.max_generate_length, opt.device)
    return tokens


def eval(**kwargs):
    opt = Config()
    for k, v in kwargs.items():  # 设置参数
        setattr(opt, k, v)

        # 数据
    dataloader = get_dataloader(opt)
    _data = dataloader.dataset._data
    word2ix, ix2word = _data['word2ix'], _data['ix2word']
    sos = word2ix.get(_data.get('sos'))
    eos = word2ix.get(_data.get('eos'))
    unknown = word2ix.get(_data.get('unknown'))
    voc_length = len(word2ix)

    # 定义模型
    encoder = EncoderRNN(opt, voc_length)
    decoder = LuongAttnDecoderRNN(opt, voc_length)

    # 加载模型
    if opt.model_ckpt == None:
        raise ValueError('model_ckpt is None.')
        return False
    checkpoint = torch.load(opt.model_ckpt, map_location=lambda s, l: s)
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    with torch.no_grad():
        # 切换模式
        encoder = encoder.to(opt.device)
        decoder = decoder.to(opt.device)
        encoder.eval()
        decoder.eval()
        # 定义seracher
        searcher = GreedySearchDecoder(encoder, decoder)

        while (1):
            input_sentence = input('> ')
            if input_sentence == 'q' or input_sentence == 'quit': break
            cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")  # 分词处理正则
            input_seq = jieba.lcut(cop.sub("", input_sentence))  # 分词序列
            input_seq = input_seq[:opt.max_input_length] + ['</EOS>']
            input_seq = [word2ix.get(word, unknown) for word in input_seq]
            tokens = generate(input_seq, searcher, sos, eos, opt)
            output_words = ''.join([ix2word[token.item()] for token in tokens])
            print('BOT: ', output_words)


def test(opt):
    # 数据
    dataloader = get_dataloader(opt)
    _data = dataloader.dataset._data
    word2ix, ix2word = _data['word2ix'], _data['ix2word']
    sos = word2ix.get(_data.get('sos'))
    eos = word2ix.get(_data.get('eos'))
    unknown = word2ix.get(_data.get('unknown'))
    voc_length = len(word2ix)

    # 定义模型
    encoder = EncoderRNN(opt, voc_length)
    decoder = LuongAttnDecoderRNN(opt, voc_length)

    # 加载模型
    if opt.model_ckpt == None:
        raise ValueError('model_ckpt is None.')
        return False
    checkpoint = torch.load(opt.model_ckpt, map_location=lambda s, l: s)
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    with torch.no_grad():
        # 切换模式
        encoder = encoder.to(opt.device)
        decoder = decoder.to(opt.device)
        encoder.eval()
        decoder.eval()
        #searcher = GreedySearchDecoder(encoder, decoder)
        # 使用效果更好的beamsearch
        searcher = BeamSearchDecoder(encoder, decoder)
        return searcher, sos, eos, unknown, word2ix, ix2word


def output_answer(input_sentence, searcher, sos, eos, unknown, opt, word2ix, ix2word):
    cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")  # 分词处理正则
    input_seq = jieba.lcut(cop.sub("", input_sentence))  # 分词序列
    input_seq = input_seq[:opt.max_input_length] + ['</EOS>']
    input_seq = [word2ix.get(word, unknown) for word in input_seq]
    tokens = generate(input_seq, searcher, sos, eos, opt)
    output_words = []
    for token in tokens:
        # 考虑到beamsearch中没有限制eos，需要在输出时提前结束
        if token.item() == eos:
            break
        output_words.append(ix2word[token.item()])


    output_words = ''.join(output_words)
    return output_words


# 使用nucleus filtering解码预测时与greedy和beam不同，单独放到一个函数
def test2(opt):
    # 数据
    dataloader = get_dataloader(opt)
    _data = dataloader.dataset._data
    word2ix, ix2word = _data['word2ix'], _data['ix2word']
    sos = word2ix.get(_data.get('sos'))
    eos = word2ix.get(_data.get('eos'))
    unknown = word2ix.get(_data.get('unknown'))
    voc_length = len(word2ix)

    # 定义模型
    encoder = EncoderRNN(opt, voc_length)
    decoder = LuongAttnDecoderRNN(opt, voc_length)

    # 加载模型
    if opt.model_ckpt == None:
        raise ValueError('model_ckpt is None.')
        return False
    checkpoint = torch.load(opt.model_ckpt, map_location=lambda s, l: s)
    encoder.load_state_dict(checkpoint['en'])
    decoder.load_state_dict(checkpoint['de'])

    with torch.no_grad():
        # 切换模式
        encoder = encoder.to(opt.device)
        decoder = decoder.to(opt.device)
        encoder.eval()
        decoder.eval()
        # 定义seracher
        #searcher = GreedySearchDecoder(encoder, decoder)
        return encoder, decoder, sos, eos, unknown, word2ix, ix2word


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        # torch.topk()返回最后一维最大的top_k个元素，返回值为二维(values,indices)
        # ...表示其他维度由计算机自行推断
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value  # 对于topk之外的其他元素的logits值设为负无穷

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)  # 对logits进行递减排序
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def filter_answer(input_sentence, encoder, decoder, sos, eos, unknown, opt, word2ix, ix2word, history):
    cop = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")  # 分词处理正则
    input_seq = jieba.lcut(cop.sub("", input_sentence))  # 分词序列
    input_seq = input_seq[:opt.max_input_length] + ['</EOS>']
    input_seq = [word2ix.get(word, unknown) for word in input_seq]

    history.append(input_seq)
    history_inputs = []
    for history_id, history_utr in enumerate(history[-opt.max_history_len:]):
        history_inputs.extend(history_utr)
        # print(history_utr)
        # print(history_id)
        #history_inputs.append(eos)

    input_batch = [history_inputs]
    for seq in input_batch:
        print(seq)
        print(len(seq))
    input_lengths = torch.tensor([len(seq) for seq in input_batch])
    input_batch = torch.LongTensor([history_inputs]).transpose(0, 1)
    input_batch = input_batch.to(opt.device)
    input_lengths = input_lengths.to(opt.device)
    #torch.Size([5, 1])
    #tensor([5])
    # Encoder的Forward计算
    print(input_batch.shape)
    print(input_lengths)
    encoder_outputs, encoder_hidden = encoder(input_batch, input_lengths)
    # 把Encoder最后时刻的隐状态作为Decoder的初始值
    decoder_hidden = encoder_hidden[:decoder.num_layers]
    # 因为我们的函数都是要求(time,batch)，因此即使只有一个数据，也要做出二维的。
    # Decoder的初始输入是SOS
    decoder_input = torch.ones(1, 1, device=opt.device, dtype=torch.long) * sos
    # 用于保存解码结果的tensor
    all_tokens = torch.zeros([0], device=opt.device, dtype=torch.long)
    response = []
    # 循环，这里只使用长度限制，后面处理的时候把EOS去掉了。
    for _ in range(opt.max_generate_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        # decoder_outputs是(batch=1, vob_size)
        next_token_logits = decoder_output[0, :]
        for id in set(response):
            next_token_logits[id] /= opt.repetition_penalty
        next_token_logits = next_token_logits / opt.temperature
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=opt.topk, top_p=opt.topp)
        # torch.multinomial表示从候选集合中无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
        all_tokens = torch.cat((all_tokens, next_token), dim=0)
        if next_token.item() == eos:  # 遇到[SEP]则表明response生成结束
            break
        response.append(next_token.item())
        decoder_input = torch.unsqueeze(next_token, 0)
    history.append(response)

    #索引转文字
    output_words = ''.join([ix2word[token.item()] for token in all_tokens if token.item() != eos])
    return output_words, history

if __name__ == "__main__":
    #import fire

    #fire.Fire()

    train()
