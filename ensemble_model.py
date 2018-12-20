import torch.nn as nn
import torch
import torch.nn.functional as F
from collections import Counter
from train_utils import get_seq_position

class EnsembleNoAttributeVote(nn.Module):
    def __init__(self, model_list, seq_max_len):
        super(EnsembleNoAttributeVote, self).__init__()
        self.models = nn.ModuleList(model_list)
        self.seq_max_len = seq_max_len

    def relpace_unk(self, preds, pred_word, unk_mask):
        sorted_preds = preds.sort(1, descending=True)[1]
        sub_preds = sorted_preds[:, 1]
        unk_index = torch.nonzero(unk_mask).squeeze(1)
        pred_word = pred_word.squeeze(1)
        pred_word[unk_index] = sub_preds[unk_index]
        pred_word = pred_word.unsqueeze(1)
        return pred_word

    def get_pred_via_prob(self, cndt_words, cndt, prob):
        word_index = {}
        for w in cndt_words:
            index = []
            for i, c in enumerate(cndt):
                if c == w:
                    index.append(i)
            word_index[w] = torch.Tensor(index).long()
        word_prob = {}
        for w in word_index.keys():
            word_prob[w] = prob.index_select(0, word_index[w]).sum().item()

        sorted_word_prob = sorted(word_prob.items(), key=lambda x: x[1], reverse=True)
        word = 9487
        for item in sorted_word_prob:
            if item[0] != 9487:
                word = item[0]
                break
        return word

    def forward(self, att_feat, fc_feat):
        att_feat = att_feat.cuda()
        fc_feat = fc_feat.cuda()
        batch_size = att_feat.size(0)

        seq = torch.Tensor([[9488]] * batch_size).long().cuda()

        with torch.no_grad():
            for _ in range(self.seq_max_len + 1):
                preds_cndt = []
                preds_prob_cndt = []

                for model in self.models:
                    model.eval()
                    preds = model(seq, att_feat, fc_feat, return_attn=False)
                    preds = preds.view(batch_size, seq.size(1), -1)[:, -1, :]

                    preds_prob, pred_word = F.softmax(preds, dim=1).max(1)
                    preds_prob = preds_prob.unsqueeze(1)
                    pred_word = pred_word.unsqueeze(1)
                    # unk_mask = pred_word.eq(9487).squeeze(1)
                    # if unk_mask.sum() != 0:
                    #     preds_prob, pred_word = self.relpace_unk(preds, pred_word, unk_mask)

                    preds_cndt.append(pred_word)
                    preds_prob_cndt.append(preds_prob)

                preds_prob_candt = torch.cat(preds_prob_cndt, dim=1).cpu()
                preds_cndt = torch.cat(preds_cndt, dim=1).tolist()
                pred_seq = []
                for i,cndt in enumerate(preds_cndt):
                    counter = Counter(cndt)
                    most_num = counter[list(counter.keys())[0]]
                    cndt_words = []
                    for key in counter:
                        if counter[key] == most_num:
                            cndt_words.append(key)

                    assert len(cndt_words) > 0

                    if len(cndt_words) == 1:
                        pred_seq.append(cndt_words[0])
                    else:

                        # prob_gather_idx = torch.Tensor([cndt.index(w) for w in cndt_words]).long()
                        # max_id = preds_prob_candt[i].index_select(0, prob_gather_idx).max(0)[1]
                        pred_seq.append(self.get_pred_via_prob(cndt_words, cndt, preds_prob_candt[i]))

                pred_seq = torch.Tensor(pred_seq).unsqueeze(1).long().cuda()
                seq = torch.cat([seq, pred_seq], dim=1)

        return seq[:, 1:]

class EnsembleNoAttributeProb(nn.Module):
    def __init__(self, model_list, seq_max_len):
        super(EnsembleNoAttributeProb, self).__init__()
        self.models = nn.ModuleList(model_list)
        self.seq_max_len = seq_max_len

    def relpace_unk(self, prob, pred_words, unk_mask):
        sorted_preds = prob.sort(1, descending=True)[1]
        sub_preds = sorted_preds[:, 1]
        unk_index = torch.nonzero(unk_mask).squeeze(1)
        pred_words[unk_index] = sub_preds[unk_index]
        return pred_words

    def get_pred_via_prob(self, cndt_words, cndt, prob):
        word_index = {}
        for w in cndt_words:
            index = []
            for i, c in enumerate(cndt):
                if c == w:
                    index.append(i)
            word_index[w] = torch.Tensor(index).long()
        word_prob = {}
        for w in word_index.keys():
            word_prob[w] = prob.index_select(0, word_index[w]).sum().item()

        sorted_word_prob = sorted(word_prob.items(), key=lambda x: x[1], reverse=True)
        word = 9487
        for item in sorted_word_prob:
            if item[0] != 9487:
                word = item[0]
                break
        return word

    def forward(self, att_feat, fc_feat):
        att_feat = att_feat.cuda()
        fc_feat = fc_feat.cuda()
        batch_size = att_feat.size(0)

        seq = torch.Tensor([[9488]] * batch_size).long().cuda()
        seq_sent = seq
        with torch.no_grad():
            for _ in range(self.seq_max_len + 1):

                preds_prob = torch.zeros([batch_size, 9490]).float().cuda()
                for model in self.models:
                    model.eval()
                    if model.method() == 'pos':
                        pos = get_seq_position(seq)
                        preds = model(seq, pos, att_feat, return_attn=False)
                    else:
                        preds = model(seq, att_feat, fc_feat, return_attn=False)
                    preds = preds.view(batch_size, seq.size(1), -1)[:, -1, :]

                    preds_prob += F.softmax(preds, dim=1)

                pred_words = preds_prob.max(1)[1]
                unk_mask = pred_words.eq(9487)

                pred_seq = pred_words.unsqueeze(1)
                seq = torch.cat([seq, pred_seq], dim=1)
                if unk_mask.sum() != 0:
                    pred_words_sent = self.relpace_unk(preds_prob, pred_words, unk_mask)
                    seq_sent = torch.cat([seq, pred_words_sent.unsqueeze(1)], dim=1)
                else:
                    seq_sent = torch.cat([seq, pred_seq], dim=1)

        return seq_sent[:, 1:]
