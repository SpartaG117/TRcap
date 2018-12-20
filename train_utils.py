import torch
import torch.nn as nn
from cider.cider import Cider
from bleu.bleu import Bleu
import torch.nn.functional as F
from spice.spice import Spice


def search(att_feat, fc_feat, model, max_len):
    model.eval()
    batch_size = att_feat.size(0)
    seq = torch.Tensor([[9488]] * batch_size).long().cuda()

    with torch.no_grad():
        for _ in range(max_len+1):
            preds = model(seq, att_feat, fc_feat, return_attn=False)
            preds = preds.view(batch_size, seq.size(1), -1)
            preds = preds[:, -1, :].max(1)[1].unsqueeze(1)
            seq = torch.cat([seq, preds], dim=1)
        preds = seq[:, 1:]
    assert preds.size(1) == max_len + 1
    return preds


def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr


def get_seq_position(seq):
    assert seq.dim() == 2
    num_ne_pad = seq.ne(0).sum(1).tolist()
    seq_pos = []
    for i in range(seq.size(0)):
        seq_pos.append([p+1 for p in range(num_ne_pad[i])] + [0] * (seq.size(1) - num_ne_pad[i]))

    if seq.is_cuda:
        return torch.Tensor(seq_pos).long().cuda()

    return torch.Tensor(seq_pos).long()


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def get_loss_correct(pred, target, criterion):
    loss = criterion(pred, target.contiguous().view(-1))
    pred = pred.max(1)[1]
    target = target.contiguous().view(-1)
    correct_mask = pred.eq(target)
    num_correct = correct_mask.masked_select(target.ne(0)).sum()
    return loss, num_correct


def get_criterion(num_vocab):
    """ make the pad loss weight 0 """
    weight = torch.ones(num_vocab).cuda()
    weight[0] = 0
    return nn.CrossEntropyLoss(weight, size_average=True)


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()

    def forward(self, log_prob, seq, rewards):

        assert seq.dim() == 2

        log_prob = log_prob.view(-1)
        no_pad_mask = seq.gt(0).view(-1)

        loss = - rewards * log_prob
        loss = loss.masked_select(no_pad_mask)
        loss = loss.mean()
        return loss


def trans2sentence(seq, vocab):
    captions = []
    if isinstance(seq, list):
        n_seq = len(seq)
    elif isinstance(seq, torch.Tensor):
        n_seq = seq.size(0)
    for i in range(n_seq):
        words = []
        for word in seq[i]:
            if word == 9489:
                break
            if word == 0:
                break
            if isinstance(seq, list):
                words.append(vocab[str(word)])
            elif isinstance(seq, torch.Tensor):
                words.append(vocab[str(word.item())])
        captions.append(' '.join(words).strip())
    return captions


def calc_cider(gts, res):

    cider = Cider()
    score, scores = cider.compute_score(gts, res)
    return score, scores


def calc_bleu(gts, res):
    bleu = Bleu()
    score, scores = bleu.compute_score(gts, res)
    return score, scores

def calc_spice(gts, res):
    spice = Spice()
    score, scores = spice.compute_score(gts, res)
    return score, scores
