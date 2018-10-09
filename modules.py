import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.models as models
import torch.nn.functional as F
import math
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()

        self.temper = math.pow(d_k, 0.5)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        """
        Args:
            q: A long Tensor of shape [num_head x batch, n, d_k]
            k: A long Tensor of shape [num_head x batch, m, d_k]
            v: A long Tensor of shape [num_head x batch, m, d_v]
            attn_mask: A ByteTensor Tensor of shape [num_head x batch, n, m]
        returns:
            output: A float Tensor of shape [num_head x batch, n, d_v]
            attn: A float Tensor of shape [num_head x batch, n, m]
        """
        # Tensor with shape [batch, n, m]
        attn = torch.bmm(q, k.transpose(1, 2))/self.temper

        # make the attention weights of padding position 0
        if attn_mask is not None:
            assert attn.size() == attn_mask.size()
            attn = attn.masked_fill(attn_mask, -float('inf'))

        # Tensor with shape [batch, n, m]
        attn = F.softmax(attn, dim=2)

        # attn dropout
        attn = self.attn_dropout(attn)

        output = torch.bmm(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_head=8, dropout=0.1):
        """
        Args:
            d_model: dimension of outputs
            num_head: number of parallel scaled dot-product attention block
            dropout:

        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_head == 0
        self.d_model = d_model
        self.num_head = num_head
        self.d_k = int(d_model/num_head)
        self.d_v = int(d_model/num_head)

        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear_out = nn.Linear(int(num_head * self.d_v), d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.w_q = nn.Parameter(torch.empty((num_head, d_model, self.d_k), dtype=torch.float))
        self.w_k = nn.Parameter(torch.empty((num_head, d_model, self.d_k), dtype=torch.float))
        self.w_v = nn.Parameter(torch.empty((num_head, d_model, self.d_v), dtype=torch.float))

        init.xavier_normal_(self.w_q)
        init.xavier_normal_(self.w_k)
        init.xavier_normal_(self.w_v)
        init.xavier_normal_(self.linear_out.weight)

    def forward(self, q, k, v, attn_mask=None):
        """
        Args:
            q: A long Tensor of shape [batch, n, d_model]
            k: A long Tensor of shape [batch, m, d_model]
            v: A long Tensor of shape [batch, m, d_model]
            attn_mask: A ByteTensor Tensor of shape [batch, n, m]

        returns:
            outputs: A float Tensor of shape [batch, n, d_model]
            attn: A ByteTensor Tensor of shape [num_head x batch, n, m]
        """
        batch_size, len_q, d_model = q.size()
        _, len_k, _ = k.size()
        _, len_v, _ = v.size()
        assert d_model == self.d_model
        assert len_k == len_v

        num_head = self.num_head
        d_k = self.d_k
        d_v = self.d_v
        residual = q

        # projections before scaled dot production attention
        # num_head x (batch_size x len_q) x d_model
        q_s = q.repeat(num_head, 1, 1).view(num_head, -1, d_model)

        # num_head x (batch_size x len_k) x d_model
        k_s = k.repeat(num_head, 1, 1).view(num_head, -1, d_model)

        # num_head x (batch_size x len_v) x d_model
        v_s = v.repeat(num_head, 1, 1).view(num_head, -1, d_model)

        q_s = torch.bmm(q_s, self.w_q).view(-1, len_q, d_k)
        k_s = torch.bmm(k_s, self.w_k).view(-1, len_k, d_k)
        v_s = torch.bmm(v_s, self.w_v).view(-1, len_v, d_v)

        # scaled dot production attention
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_head, 1, 1)
        outputs, attn = self.attention(q_s, k_s, v_s, attn_mask=attn_mask)

        # Concat multihead attention, outputs size = batch_size * len_q * (h * d_v)
        outputs = torch.cat(torch.split(outputs, batch_size, dim=0), dim=-1)

        # final linear
        outputs = self.linear_out(outputs)

        #dropout
        outputs = self.dropout(outputs)

        # layer normalization
        outputs = self.layer_norm(outputs + residual)

        return outputs, attn

class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)

        self.feed_1 = nn.Conv1d(d_model, d_ff, 1)
        self.feed_2 = nn.Conv1d(d_ff, d_model, 1)
        self.layer_norm = nn.LayerNorm(d_model)

        # init weights
        # init.kaiming_normal_(self.feed_1.weight, mode='fan_in', nonlinearity='relu')
        init.xavier_normal_(self.feed_1.weight)
        init.xavier_normal_(self.feed_2.weight)
        init.constant_(self.feed_1.bias, 0)
        init.constant_(self.feed_2.bias, 0)

    def forward(self, x):

        residual = x

        outputs = self.feed_1(x.transpose(1, 2))
        outputs = F.relu(outputs)
        outputs = self.feed_2(outputs).transpose(1, 2)
        outputs = self.dropout(outputs)

        outputs = self.layer_norm(outputs + residual)

        return outputs




class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_head, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_head, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_head, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, dec_input, enc_output, self_attn_mask=None):
        outputs, dec_self_attn = self.self_attn(dec_input, dec_input, dec_input, self_attn_mask)
        outputs, cross_attn = self.cross_attn(outputs, enc_output, enc_output, None)
        outputs = self.feed_forward(outputs)
        return outputs, dec_self_attn, cross_attn


class CNNEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(CNNEncoder, self).__init__()

        self.proj_avg = nn.Linear(2048, d_model)
        self.proj_spa = nn.Linear(2048, d_model)
        self.avg_pool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(dropout)

        # init weight
        # init.kaiming_normal_(self.proj_1.weight, mode='fan_in', nonlinearity='relu')
        init.xavier_normal_(self.proj_avg.weight)
        init.constant_(self.proj_avg.bias, 0)
        # init.kaiming_normal_(self.proj_2.weight, mode='fan_in', nonlinearity='relu')
        init.xavier_normal_(self.proj_spa.weight)
        init.constant_(self.proj_spa.bias, 0)

    def forward(self, feats):

        # res_feats: tensor with shape B x 2048 x 7 x 7
        feats_avg = self.avg_pool(feats).view(feats.size(0), feats.size(1))  # B x 2048
        feats_avg = self.proj_avg(feats_avg)  # B x 512
        feats_avg = F.relu(feats_avg)
        feats_avg = self.dropout(feats_avg)

        feats_spa = feats.view(feats.size(0), 2048, -1).transpose(1, 2)  # B x 49 x 2048
        feats_spa = self.proj_spa(feats_spa)  # B x 49 x 512
        feats_spa = F.relu(feats_spa)
        feats_spa = self.dropout(feats_spa)

        return feats_spa, feats_avg



def padding_attn_mask(seq_q, seq_k):

    assert seq_q.dim() == seq_k.dim()
    assert seq_q.dim() == 2 and seq_k.dim() == 2
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # b x 1 x len(seq_k)
    pad_attn_mask = seq_k.eq(0).unsqueeze(1)
    pad_attn_mask = pad_attn_mask.expand(batch_size, len_q, len_k)
    if seq_q.is_cuda:
        return pad_attn_mask.cuda()
    return pad_attn_mask

def sequence_attn_mask(seq_q):

    assert seq_q.dim() == 2
    shape = (seq_q.size(0), seq_q.size(1), seq_q.size(1))

    # torch.triu can not deal with batch input
    seq_attn_mask = np.triu(np.ones(shape), k=1).astype('uint8')
    seq_attn_mask = torch.from_numpy(seq_attn_mask)

    if seq_q.is_cuda:
        return seq_attn_mask.cuda()
    return seq_attn_mask



if __name__ == '__main__':
    model = Decoder(10050,512,512,6,8,2048,0.1)
    #model = CNNEncoder(512)
