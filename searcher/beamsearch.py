# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np


class BeamSearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(BeamSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, sos, eos, input_seq, input_length, max_length, device):
        # Encoder的Forward计算
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # 把Encoder最后时刻的隐状态作为Decoder的初始值
        decoder_hidden = encoder_hidden[:self.decoder.num_layers]
        # 因为我们的函数都是要求(time,batch)，因此即使只有一个数据，也要做出二维的。
        # Decoder的初始输入是SOS
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * sos
        # 用于保存解码结果的tensor(这里已经固定了beamSize=3，所以直接创建了三个变量便于操作)
        all_token1 = torch.zeros([0], device=device, dtype=torch.long)
        all_token2 = torch.zeros([0], device=device, dtype=torch.long)
        all_token3 = torch.zeros([0], device=device, dtype=torch.long)
        # 初始三条束得分均为1
        scores = torch.ones(1, 3)
        # Decoder forward一步
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden,
                                                      encoder_outputs)
        decoder_hidden_branch1 = decoder_hidden_branch2 = decoder_hidden_branch3 = decoder_hidden
        decoder_scores, decoder_input = torch.topk(decoder_output, k=3, dim=1)
        token1, token2, token3 = torch.chunk(decoder_input, 3, dim=1)
        all_token1 = torch.cat((all_token1, token1), dim=0)
        all_token2 = torch.cat((all_token2, token2), dim=0)
        all_token3 = torch.cat((all_token3, token3), dim=0)
        scores *= decoder_scores
        for _ in range(max_length):
            decoder_output_branch1, decoder_hidden_branch1 = self.decoder(token1, decoder_hidden_branch1,
                                                                          encoder_outputs)
            decoder_output_branch2, decoder_hidden_branch2 = self.decoder(token2, decoder_hidden_branch2,
                                                                          encoder_outputs)
            decoder_output_branch3, decoder_hidden_branch3 = self.decoder(token3, decoder_hidden_branch3,
                                                                          encoder_outputs)

            decoder_scores_branch1, decoder_input_branch1 = torch.topk(decoder_output_branch1, k=3, dim=1)
            decoder_scores_branch2, decoder_input_branch2 = torch.topk(decoder_output_branch2, k=3, dim=1)
            decoder_scores_branch3, decoder_input_branch3 = torch.topk(decoder_output_branch3, k=3, dim=1)

            decoder_input_list_temp = torch.cat((decoder_input_branch1, decoder_input_branch2, decoder_input_branch3),
                                                dim=1)
            scores_temp = torch.cat((decoder_scores_branch1 * scores[0][0], decoder_scores_branch2 * scores[0][1],
                                     decoder_scores_branch3 * scores[0][2]), dim=1)
            scores_temp = torch.squeeze(scores_temp, 0)
            score_list = scores_temp.detach().numpy().tolist()
            temp = score_list.copy()  # 复制一份，不破坏原来的列表
            temp.sort()
            max1 = temp[-1]
            max2 = temp[-2]
            max3 = temp[-3]
            index1 = score_list.index(max1)
            index2 = score_list.index(max2)
            index3 = score_list.index(max3)
            # 更新下一轮token输入(需要增加两个冗余维度保持数据格式一致)
            token1 = decoder_input_list_temp[0][index1].unsqueeze(0).unsqueeze(0)
            token2 = decoder_input_list_temp[0][index2].unsqueeze(0).unsqueeze(0)
            token3 = decoder_input_list_temp[0][index3].unsqueeze(0).unsqueeze(0)
            # 更新三条beams得分
            score_l = [score_list[index1], score_list[index2], score_list[index3]]
            score_np = np.array(score_l)
            score_t = torch.from_numpy(score_np)
            scores = score_t.unsqueeze(0)
            # 记录token序列
            all_token1 = torch.cat((all_token1, token1), dim=0)
            all_token2 = torch.cat((all_token2, token2), dim=0)
            all_token3 = torch.cat((all_token3, token3), dim=0)
        # 返回最终最高得分
        # print(scores[0][0])
        return all_token1, scores[0][0]
