from .modules import *

class TransformerCap(nn.Module):
    def __init__(self, num_vocab, d_word, d_model, num_layer, num_head, d_ff, dropout=0.1):
        super(TransformerCap, self).__init__()
        self.img_encoder = CNNEncoder(d_model,  num_head, d_ff, num_layer, dropout)
        # self.attr_encoder = AttributeEncoder(d_model, num_head, d_ff, num_layer, dropout)
        self.word_embedding = nn.Embedding(num_vocab, d_word, padding_idx=0)

        self.decoder_stack = nn.ModuleList([DecoderLayer(d_model, num_head, d_ff, dropout)
                                            for _ in range(num_layer)])

        self.proj_fc = nn.Linear(2048, d_model)
        self.res_proj = nn.Linear(2*d_model, d_model)
        self.out_proj = nn.Linear(d_model, num_vocab, bias=False)
        self.dropout = nn.Dropout(dropout)

        # init weight
        init.xavier_normal_(self.proj_fc.weight)
        init.constant_(self.proj_fc.bias, 0)
        init.xavier_normal_(self.word_embedding.weight)
        init.xavier_normal_(self.res_proj.weight)
        init.xavier_normal_(self.out_proj.weight)
        self.res_proj.bias.data.zero_()

    def forward(self, seq, att_feat, fc_feat, return_attn=False):
        dec_pad_attn_mask = padding_attn_mask(seq, seq)
        dec_seq_attn_mask = sequence_attn_mask(seq)
        dec_self_attn_mask = torch.gt(dec_pad_attn_mask + dec_seq_attn_mask, 0)

        word_embed = self.word_embedding(seq)

        # res_feats: tensor with shape B x 2048 x 7 x 7
        fc_feats = self.proj_fc(fc_feat)  # B x 512
        fc_feats = F.relu(fc_feats, inplace=True)
        fc_feats = self.dropout(fc_feats)

        fc_feats = fc_feats.unsqueeze(1).expand_as(word_embed)

        att_feats = self.img_encoder(att_feat)  # B x 49 x 512 , B x 512
        # attr_feats = self.attr_encoder(attr_tp_feats)

        x = torch.cat([word_embed, fc_feats], dim=2)
        x = self.res_proj(x)
        x = self.dropout(x)

        if return_attn:
            dec_self_attn = []
            dec_cross_attn_bt = []
            # dec_cross_attn_attr = []

        output = x
        for dec_layer in self.decoder_stack:
            output, self_attn, cross_attn_bt = dec_layer(output, att_feats, dec_self_attn_mask)
            if return_attn:
                dec_self_attn += [self_attn]
                dec_cross_attn_bt += [cross_attn_bt]
                # dec_cross_attn_attr += [cross_attn_attr]
        output = self.out_proj(output)

        if return_attn:
            return output.view(-1, output.size(2)), dec_self_attn, dec_cross_attn_bt
        else:
            return output.view(-1, output.size(2))

    def method(self):
        return 'no_pos1'
