from modules import *

class ResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet, self).__init__()
        resnet = models.resnet152(pretrained)
        module = list(resnet.children())[:-2]
        self.resnet_conv = nn.Sequential(*module)

    def forward(self, imgs):
        # tensor with shape B x 2048 x 7 x 7
        feats = self.resnet_conv(imgs)
        return feats


class TransformerCap(nn.Module):
    def __init__(self, num_vocab, d_word, d_model, num_layer, num_head, d_ff, dropout=0.1):
        super(TransformerCap, self).__init__()
        self.img_encoder = CNNEncoder(d_model, dropout)
        self.word_embedding = nn.Embedding(num_vocab, d_word, padding_idx=0)

        self.decoder_stack = nn.ModuleList([DecoderLayer(d_model, num_head, d_ff, dropout)
                                            for _ in range(num_layer)])

        self.res_proj = nn.Linear(2*d_model, d_model)
        self.out_proj = nn.Linear(d_model, num_vocab, bias=False)
        self.dropout = nn.Dropout(dropout)

        # init weight
        init.xavier_normal_(self.word_embedding.weight)
        init.xavier_normal_(self.res_proj.weight)
        init.xavier_normal_(self.out_proj.weight)
        self.res_proj.bias.data.zero_()

    def forward(self, seq, feats, return_attn=False):
        dec_pad_attn_mask = padding_attn_mask(seq, seq)
        dec_seq_attn_mask = sequence_attn_mask(seq)
        dec_self_attn_mask = torch.gt(dec_pad_attn_mask + dec_seq_attn_mask, 0)

        word_embed = self.word_embedding(seq)
        feats_spa, feats_avg = self.img_encoder(feats)  # B x 49 x 512 , B x 512
        feats_avg = feats_avg.unsqueeze(1).expand_as(word_embed)

        x = torch.cat([word_embed, feats_avg], dim=2)
        x = self.res_proj(x)
        x = self.dropout(x)

        if return_attn:
            dec_self_attn = []
            dec_cross_attn = []

        output = x
        for dec_layer in self.decoder_stack:
            output, self_attn, cross_attn = dec_layer(output, feats_spa, dec_self_attn_mask)
            if return_attn:
                dec_self_attn += [self_attn]
                dec_cross_attn += [cross_attn]

        output = self.out_proj(output)

        if return_attn:
            return output.view(-1, output.size(2)), dec_self_attn, dec_cross_attn
        else:
            return output.view(-1, output.size(2))
