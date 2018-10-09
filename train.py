import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import time
import pickle as pk
import os
import argparse
from Model import TransformerCap, ResNet
from coco_loader_v1 import CocoDataset, collate_fn, Vocabulary
from pycocotools.coco import COCO
from tokenizer.ptbtokenizer import PTBTokenizer
from cider.cider import Cider


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
    return nn.CrossEntropyLoss(weight, reduction='elementwise_mean')


class RewardCriterion(nn.Module):
    def __init__(self):
        super(RewardCriterion, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, outputs, seq, rewards):

        # rewards is a list of [b]
        assert seq.dim() == 2
        pad_mask = seq.ge(1).view(-1)
        out_index = seq.contiguous().view(-1).unsqueeze(1)

        log_prob = self.log_softmax(outputs)


        # tensor of shape b x (seq_len - 1) x 1
        log_prob = torch.gather(log_prob, 1, out_index).contiguous().view(-1)

        loss = rewards * log_prob
        loss = loss.masked_select(pad_mask)
        loss = loss.mean()
        return loss


def search(feats, model, max_len):
    batch_size = feats.size(0)
    seq = torch.ones([batch_size, 1]).long().cuda()
    for _ in range(max_len):
        preds = model(seq, feats, return_attn=False)
        preds = preds.view(batch_size, seq.size(1), -1)
        preds = preds[:, -1, :].max(1)[1].unsqueeze(1)
        seq = torch.cat([seq, preds], dim=1)
    preds = seq[:, 1:]
    assert preds.size(1) == max_len
    return preds

def greedy_sample(feats, model, max_len):
    softmax = nn.Softmax(dim=1).cuda()
    batch_size = feats.size(0)
    seq = torch.ones([batch_size, 1]).long().cuda()
    for _ in range(max_len + 1):
        preds = model(seq, feats, return_attn=False)
        preds = preds.view(batch_size, seq.size(1), -1)
        prob = softmax(preds[:, -1, :])
        samples = torch.multinomial(prob, 1)
        seq = torch.cat([seq, samples], dim=1)
    sample_w = seq
    assert sample_w.size(1) == max_len + 2

    # TODO: if there is a more effective way
    for i in range(batch_size):
        flag = 0
        for j in range(max_len + 1):
            if flag == 1:
                sample_w[i, j] = 0
            if sample_w[i, j] == 2:
                flag = 1
    return sample_w


def tensor2sentence(seq, vocab):
    captions = []
    for i in range(seq.size(0)):
        words = []
        for word in seq[i]:
            if word == 2:
                break
            words.append(vocab.id2word[word.item()])
        captions.append(' '.join(words))
    return captions

def prepro_gts(data):
    gts = {}
    for batch in data:
        _, _, _, captions, img_ids = batch
        for i, img_id in enumerate(img_ids):
            gt = []
            for cap in captions[i]:
                cap_dict = {}
                cap_dict['image_id'] = img_ids
                cap_dict['caption'] = cap
                gt.append(cap_dict)
            gts[img_id] = gt
    return gts

def calc_cider(gts, res):
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)
    cider = Cider()
    score, scores = cider.compute_score(gts, res)
    return score, scores

def train_epoch(model, res_model, train_data, criterion, optimizer, img_optimizer):
    """ training epoch"""

    model.train()
    total_loss = 0
    total_words = 0
    total_correct = 0
    start = time.time()
    for idx, batch in tqdm(enumerate(train_data), mininterval=2,
            desc='  - (Training)   ', leave=False, total=len(train_data)):

        imgs, src_seq, tgt_seq, captions, img_ids = batch
        imgs = imgs.cuda()

        # Tensor of shape [B x 1 x 2048 x 7 x 7]
        feats = res_model(imgs).unsqueeze(1)
        feats = feats.expand(feats.size(0), 5, feats.size(2), feats.size(3), feats.size(4))

        # Tensor of shape [(B x 5) x 2048 x 7 x 7]
        feats = feats.contiguous().view(feats.size(0)*feats.size(1), feats.size(2), feats.size(3), feats.size(4))

        # both Tensor of shape [(B x 5) x 19]
        src_seq = src_seq.view(src_seq.size(0)*src_seq.size(1), src_seq.size(2)).cuda()
        tgt_seq = tgt_seq.view(tgt_seq.size(0)*tgt_seq.size(1), tgt_seq.size(2)).cuda()

        optimizer.zero_grad()
        if img_optimizer:
            img_optimizer.zero_grad()

        pred = model(src_seq, feats, return_attn=False)

        num_words = tgt_seq.contiguous().ne(0).sum()
        loss, correct = get_loss_correct(pred, tgt_seq, criterion)
        loss.backward()
        optimizer.step()
        if img_optimizer:
            img_optimizer.step()

        total_loss += loss.item()
        total_words += num_words.item()
        total_correct += correct.item()

        # if idx == 10:
        #     break

    duration = (time.time() - start) / 60
    print("Time Consuming %3.2f" % duration)

    torch.cuda.empty_cache()
    return total_loss/len(train_data), total_correct/total_words


def self_critical_train_epoch(model, res_model, train_data, rl_criterion, optimizer, vocab, max_len):
    """ self-critical reinforcement learning training epoch"""

    total_loss = 0

    for idx, batch in tqdm(enumerate(train_data), mininterval=2, desc='  -(RL fine-tune)  ', \
                           leave=False, total=len(train_data)):

        gts = prepro_gts([batch])
        imgs, _, _, captions, img_ids = batch
        imgs = imgs.cuda()

        # Tensor of shape [B x 2048 x 7 x 7]
        feats = res_model(imgs)

        # sample step
        model.eval()
        with torch.no_grad():
            sample_w = greedy_sample(feats, model, max_len)
            sample_sentence = tensor2sentence(sample_w[:, 1:-1], vocab)
            baseline_w = search(feats, model, max_len)
            baseline_sentence = tensor2sentence(baseline_w, vocab)
            sample_res = {}
            baseline_res = {}
            for i, img_id in enumerate(img_ids):
                s_cap_dict = {}
                s_cap_dict['image_id'] = img_id
                s_cap_dict['caption'] = sample_sentence[i]
                sample_res[img_id] = [s_cap_dict]

                b_cap_dict = {}
                b_cap_dict['image_id'] = img_id
                b_cap_dict['caption'] = baseline_sentence[i]
                baseline_res[img_id] = [b_cap_dict]

        _, s_rewards = calc_cider(gts, sample_res)
        _, b_rewards = calc_cider(gts, baseline_res)

        s_rewards = torch.tensor(s_rewards).float().unsqueeze(1).expand(sample_w.size(0), sample_w[:, :-1].size(1))
        s_rewards = s_rewards.contiguous().view(-1)
        b_rewards = torch.tensor(b_rewards).float().unsqueeze(1).expand(sample_w.size(0), sample_w[:, :-1].size(1))
        b_rewards = b_rewards.contiguous().view(-1)
        rewards = b_rewards - s_rewards
        rewards = rewards.cuda()

        optimizer.zero_grad()
        model.train()
        outputs = model(sample_w[:, :-1], feats, return_attn=False)
        loss = rl_criterion(outputs, sample_w[:, 1:], rewards)

        total_loss += loss.item()

        loss.backward()
        optimizer.step()


    return total_loss / len(train_data)


def eval_epoch(model, res_model, val_data, vocab, max_len, gts):
    """ evalution epoch """

    model.eval()
    res = {}
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_data), mininterval=2,
                              desc='  -(evalution)  ', leave=False, total=len(val_data)):

            imgs, _, _, _, img_ids = batch
            imgs = imgs.cuda()

            # Tensor of shape [B x 2048 x 7 x 7]
            feats = res_model(imgs)
            preds = search(feats, model, max_len)
            pred_cap = tensor2sentence(preds, vocab)

            for i, img_id in enumerate(img_ids):
                cap_dict = {}
                cap_dict['image_id'] = img_id
                cap_dict['caption'] = pred_cap[i]
                res[img_id] = [cap_dict]

    cider_score, _ = calc_cider(gts, res)
    return cider_score


def train(model, res_model, train_data, val_data, criterion, optimizer, img_optimizer,
          current_epoch, args):

    epoch = current_epoch + 1

    if args.log_path:
        log_train_file = args.log_path + 'train.log'
        log_valid_file = args.log_path + 'valid.log'
    else:
        log_train_file = None
        log_valid_file = None

    if epoch == 1:
        with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
            log_tf.write('epoch,loss,accuracy\n')
            log_vf.write('epoch,cider\n')

    rl_criterion = RewardCriterion().cuda()

    with open(os.path.join(args.data_path, args.vocab), 'rb') as f:
        vocab = pk.load(f)
    gts = prepro_gts(val_data)
    val_scores = []

    while(epoch <= args.epoch):
        print('[ Epoch ', epoch, ']')
        start = time.time()

        if epoch >= args.finetune_epoch and img_optimizer is None:
            for param in res_model.parameters():
                param.requires_grad = True
            img_optimizer = optim.Adam(res_model.parameters(), 1e-5)
            # img_scheduler = lr_scheduler.StepLR(img_optimizer, step_size=10, gamma=0.1)

        # scheduler.step()
        # if epoch >= args.finetune_epoch and img_scheduler:
        #     img_scheduler.step()

        if epoch >= args.rl_finetune_epoch:
            train_loss = self_critical_train_epoch(model, res_model, train_data, rl_criterion, optimizer, \
                                                   vocab, args.max_length)

            print('  - (Training)   loss: {loss: 3.5f}, elapse: {elapse:3.3f} min'.format(
                      loss=train_loss, elapse=(time.time()-start)/60))
        else:

            train_loss, train_acc = train_epoch(model, res_model, train_data, criterion, optimizer, img_optimizer)

            print('  - (Training)   loss: {loss: 3.5f}, accuracy: {acc:3.3f} %, '\
                  'elapse: {elapse:3.3f} min'.format(
                      loss=train_loss, acc=100*train_acc,
                      elapse=(time.time()-start)/60))

        val_score = eval_epoch(model, res_model, val_data, vocab, args.max_length, gts)

        print('  - (Validation)  cider: {score: 3.3f}, elapse: {elapse:3.3f} min'.format(
                    score=val_score, elapse=(time.time()-start)/60))

        val_scores += [val_score]

        model_state_dict = model.state_dict()
        res_model_state_dict = res_model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'res_model': res_model_state_dict,
            'settings': args,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            # 'scheduler': scheduler.state_dict(),
            'img_optimizer': img_optimizer.state_dict() if img_optimizer else None,
            # 'img_scheduler': img_scheduler.state_dict() if img_scheduler else None,
        }

        if args.checkpoint_dir:
            # if args.save_mode == 'all':
            ckpt_path = args.checkpoint_dir + 'epoch_{epoch}_score_{score:3.3f}.ckpt'.format(epoch=epoch, score=val_score)
            torch.save(checkpoint, ckpt_path)
            if args.save_mode == 'best':
                ckpt_path = args.checkpoint_dir + 'best.ckpt'
                if val_score >= max(val_scores):
                    torch.save(checkpoint, ckpt_path)
                    print('    - [Info] The checkpoint file has been updated.')

            ckpt_path = args.checkpoint_dir + 'latest.ckpt'
            torch.save(checkpoint, ckpt_path)
            print('    - [Info] The checkpoint file has been updated.')

        if log_train_file and log_valid_file:
            with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
                log_tf.write('{epoch},{loss: 8.5f},{acc:3.3f}\n'.format(
                    epoch=epoch, loss=train_loss, acc=100*train_acc))
                log_vf.write('{epoch},{score:3.3f}\n'.format(
                    epoch=epoch, score=val_score))

        epoch += 1

def main(args):

    #================== Loading Data ======================#

    train_dataset = CocoDataset(
        root=args.data_path,
        json_path=args.train_json,
        vocab_path=args.vocab,
        max_length=args.max_length,
        phase='train')

    val_dataset = CocoDataset(
        root=args.data_path,
        json_path=args.val_json,
        vocab_path=args.vocab,
        max_length=args.max_length,
        phase='val')

    train_data = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn)

    val_data = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn)

    #=================== Establish model====================#

    model = TransformerCap(
        num_vocab=args.num_vocab,
        d_word=args.d_word_vec,
        d_model=args.d_model,
        num_layer=args.num_layer,
        num_head=args.num_head,
        d_ff=args.d_ff,
        dropout=args.dropout)

    res_model = ResNet()

    if args.cuda:
        model.cuda()
        res_model.cuda()

    #==================== Loading checkpoint ========================#

    checkpoint_path = args.checkpoint_dir+'best.ckpt'
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        res_model.load_state_dict(checkpoint['res_model'])
        current_epoch = checkpoint['epoch']
    else:
        current_epoch = 0

    #===================== criterion and optimizer ===================#

    if current_epoch + 1 < args.finetune_epoch:
        for param in res_model.parameters():
            param.requires_grad = False
    else:
        for param in res_model.parameters():
            param.requires_grad = True

    criterion = get_criterion(args.num_vocab)
    optimizer = optim.Adam(model.parameters(), 1e-5)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=.1)

    if current_epoch + 1 >= args.finetune_epoch:
        img_optimizer = optim.Adam(res_model.parameters(), 1e-5)
        # img_scheduler = lr_scheduler.StepLR(img_optimizer, step_size=args.lr_step_size, gamma=0.1)
    else:
        img_optimizer = None
        # img_scheduler = None

    if os.path.isfile(checkpoint_path) and current_epoch > 0:
        optimizer.load_state_dict(checkpoint['optimizer'])
        # scheduler.load_state_dict(checkpoint['scheduler'])
        if checkpoint['img_optimizer'] and checkpoint['img_scheduler']:
            img_optimizer.load_state_dict(checkpoint['img_optimizer'])
            # img_scheduler.load_state_dict(checkpoint['img_scheduler'])

    #==================== Training ========================#

    # optimizer = optim.SGD(model.__get_nonconv_param__(), lr=1e-7)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=.1)
    # img_optimizer = optim.SGD(model.__get_conv_param__(), lr=1e-7)
    # img_scheduler = lr_scheduler.StepLR(img_optimizer, step_size=5, gamma=0.1)

    train(model, res_model, train_data, val_data, criterion, optimizer,
          img_optimizer, current_epoch, args)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data path
    parser.add_argument('-data_path', type=str, default='./data/')
    parser.add_argument('-train_json', type=str, default='coco_train.json')
    parser.add_argument('-val_json', type=str, default='coco_val.json')
    parser.add_argument('-vocab', type=str, default='vocab.pkl')

    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-max_length', type=int, default=18)

    parser.add_argument('-num_head', type=int, default=8)
    parser.add_argument('-num_layer', type=int, default=6)
    parser.add_argument('-num_vocab', type=int, default=10054)
    parser.add_argument('-d_word_vec', type=int, default=512)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-d_ff', type=int, default=1024)

    #parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-dropout', type=float, default=0.1)

    parser.add_argument('-finetune_epoch', type=int, default=15)
    parser.add_argument('-rl_finetune_epoch', type=int, default=15)

    parser.add_argument('-lr_step_size', type=int, default=15)

    #parser.add_argument('-embs_share_weight', action='store_true')
    #parser.add_argument('-proj_share_weight', action='store_true')

    parser.add_argument('-log_path', type=str, default='./log/')
    parser.add_argument('-checkpoint_dir', type=str, default='./checkpoints/')
    parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')
    parser.add_argument('--cuda', type=bool, default=True)

    args = parser.parse_args()

    main(args)

