from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import time
import os
from Model_enc_no_attr_fc_1.Model import TransformerCap
from coco_loader_with_val import CocoDataset, collate_fn
import opts
from ciderD.ciderD import CiderD
from misc.reward import *
from train_utils import *
import json

cider_D = CiderD(df='coco-train-words')


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


def greedy_sample(att_feat, fc_feat, model, max_len):
    model.eval()
    batch_size = att_feat.size(0)
    seq = torch.Tensor([[9488]] * batch_size).long().cuda()

    zero_mask = torch.zeros([batch_size]).byte().cuda()
    end_tokens = torch.Tensor([9489 for _ in range(batch_size)]).long().cuda()
    pad_tokens = torch.Tensor([0 for _ in range(batch_size)]).long().cuda()
    start_tokens = torch.Tensor([9488 for _ in range(batch_size)]).long().cuda()

    with torch.no_grad():
        for i in range(max_len + 1):
            preds = model(seq, att_feat, fc_feat, return_attn=False)
            preds = preds.view(batch_size, seq.size(1), -1)
            prob = F.softmax(preds[:, -1, :], dim=1)
            samples = torch.multinomial(prob, 1).squeeze(1)
            samples = samples.masked_fill(zero_mask, 0)

            end_mask = samples.eq(end_tokens)
            start_mask = samples.eq(start_tokens)
            pad_mask = samples.eq(pad_tokens)
            zero_mask = (end_mask + start_mask + pad_mask).ge(1)

            seq = torch.cat([seq, samples.unsqueeze(1)], dim=1)

    sample_tgt = seq[:, 1:]
    sample_src = seq[:, :-1]

    return sample_src, sample_tgt


def prepro_gtsD(data, vocab):
    gts = {}
    for batch in data:
        _, _, _, _, infos = batch
        for i, info in enumerate(infos):
            caps = trans2sentence(info['gts'], vocab)
            gts[i] = caps
    return gts


def train_epoch(model, train_data, criterion, optimizer):
    """ training epoch"""

    model.train()
    total_loss = 0
    total_words = 0
    total_correct = 0
    start = time.time()
    for idx, batch in tqdm(enumerate(train_data), mininterval=2,
            desc='  - (Training)   ', leave=False, total=len(train_data)):

        src_seq, tgt_seq,  att_feat, fc_feat, _ = batch

        att_feat = att_feat.unsqueeze(1).expand(-1, src_seq.size(1), -1, -1)
        fc_feat = fc_feat.unsqueeze(1).expand(-1, src_seq.size(1), -1)
        att_feat = att_feat.contiguous().view(att_feat.size(0)*att_feat.size(1), att_feat.size(2), -1).cuda()
        fc_feat = fc_feat.contiguous().view(fc_feat.size(0)*fc_feat.size(1), -1).cuda()

        # both Tensor of shape [(B x 5) x 17]
        src_seq = src_seq.view(src_seq.size(0)*src_seq.size(1), src_seq.size(2)).cuda()
        tgt_seq = tgt_seq.view(tgt_seq.size(0)*tgt_seq.size(1), tgt_seq.size(2)).cuda()

        optimizer.zero_grad()

        pred = model(src_seq, att_feat, fc_feat, return_attn=False)

        num_words = tgt_seq.contiguous().ne(0).sum()
        loss, correct = get_loss_correct(pred, tgt_seq, criterion)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_words += num_words.item()
        total_correct += correct.item()

    duration = (time.time() - start) / 60
    print("Time Consuming %3.2f" % duration)

    torch.cuda.empty_cache()

    return total_loss/len(train_data), total_correct/total_words


def self_critical_train_epoch(model, train_data, rl_criterion, optimizer, vocab, max_len, opt):
    """ self-critical reinforcement learning training epoch"""

    total_loss = 0
    total_reward = 0

    for idx, batch in tqdm(enumerate(train_data), mininterval=2, desc='  -(RL fine-tune)  ', \
                           leave=False, total=len(train_data)):

        gts = prepro_gtsD([batch], vocab)

        _, _,  att_feat, fc_feat, infos = batch

        batch_size = att_feat.size(0)
        att_feat = att_feat.cuda()
        fc_feat = fc_feat.cuda()

        # sample step
        baseline = search(att_feat, fc_feat, model, max_len)
        baseline_sentence = trans2sentence(baseline, vocab)

        optimizer.zero_grad()

        sample_src, sample_tgt = greedy_sample(att_feat, fc_feat, model, max_len)
        sample_sentence = trans2sentence(sample_tgt, vocab)

        rewards, avg_rewards = get_self_critical_reward_with_bleu(baseline_sentence,
                                                                sample_sentence,
                                                                gts,
                                                                batch_size,
                                                                max_len+1,
                                                                cider_D,
                                                                opt)

        total_reward += avg_rewards

        model.eval()
        outputs = model(sample_src, att_feat, fc_feat, return_attn=False)
        log_prob = F.log_softmax(outputs, dim=1).view(batch_size, max_len+1, -1)
        log_prob = log_prob.gather(2, sample_tgt.unsqueeze(2))

        loss = rl_criterion(log_prob, sample_tgt, rewards)

        print('iter', idx, 'avg_rewards', avg_rewards, 'loss:', loss.item())

        total_loss += loss.item()

        loss.backward()
        # clip_gradient(optimizer, 0.1)
        optimizer.step()
        #
        # if idx == 100:
        #     break

    return total_loss, total_reward


def eval_epoch(model, val_data, vocab, max_len, gts):
    """ evalution epoch """

    model.eval()
    res = {}
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_data), mininterval=2,
                              desc='  -(evalution)  ', leave=False, total=len(val_data)):

            _, _, att_feat, fc_feat, infos = batch

            att_feat = att_feat.squeeze(1).cuda()
            fc_feat = fc_feat.squeeze(1).cuda()

            # Tensor of shape [B x 2048 x 7 x 7]
            preds = search(att_feat, fc_feat, model, max_len)
            pred_cap = trans2sentence(preds, vocab)

            for i, info in enumerate(infos):
                res[str(info['image_id'])] = [pred_cap[i]]

    cider_score, _ = calc_cider(gts, res)
    bleu_score, _ = calc_bleu(gts, res)
    spice_score, _ = calc_spice(gts, res)
    return cider_score, bleu_score[3], spice_score


def train(model, train_data, val_data, criterion, optimizer, vocab, current_epoch, args):

    epoch = current_epoch + 1

    gts = json.load(open('data/eval_gts.json', 'r'))

    while epoch <= args.epoch:
        print('[ Epoch ', epoch, ']')
        start = time.time()

        if epoch >= args.rl_finetune_epoch or args.rl_criterion_flag:
            if epoch == args.rl_finetune_epoch:
                optimizer = optim.Adam(model.parameters(), 1e-5)


            ################ lr decay ###################
            # frac = (epoch - args.rl_finetune_epoch) // args.rl_finetune_epoch
            # decay_factor = 0.6 ** frac
            # args.current_lr = 1e-5 * decay_factor
            #
            # set_lr(optimizer, args.current_lr)
            rl_criterion = RewardCriterion().cuda()
            train_loss, reward = self_critical_train_epoch(model, train_data, rl_criterion, optimizer, \
                                                   vocab, args.max_length, args)

            print('  - (RL Training)   loss: {loss: 3.5f}, avg_reward: {avg_reward: 3.5f}, elapse: {elapse:3.3f} min'.format(
                      loss=train_loss, avg_reward=reward,  elapse=(time.time()-start)/60))
        else:

            train_loss, train_acc = train_epoch(model,  train_data, criterion, optimizer)

            print('  - (Training)   loss: {loss: 3.5f}, accuracy: {acc:3.3f} %, '\
                  'elapse: {elapse:3.3f} min'.format(
                      loss=train_loss, acc=100*train_acc,
                      elapse=(time.time()-start)/60))

        cider_score, bleu_socre, spice_score = eval_epoch(model, val_data, vocab, args.max_length, gts)

        print('  - (Validation)  cider: {cider: 3.4f}, bleu4: {bleu: 3.4f}, spice: {spice: 3.4f}, elapse: {elapse:3.3f} min'.format(
                    cider=cider_score, bleu=bleu_socre, spice=spice_score, elapse=(time.time()-start)/60))

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': args,
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
        }

        if args.checkpoint_dir:
            ckpt_path = args.checkpoint_dir + 'epoch_{epoch}_cider_{cider:3.4f}_bleu4_{bleu:3.4f}_spice_{spice: 3.4f}.ckpt'.format(
                        epoch=epoch, cider=cider_score, bleu=bleu_socre, spice=spice_score)
            torch.save(checkpoint, ckpt_path)

            ckpt_path = args.checkpoint_dir + 'latest.ckpt'
            torch.save(checkpoint, ckpt_path)
            print('    - [Info] The checkpoint file has been updated.')

        epoch += 1

def main(args):


    #=================== Establish model====================#

    model = TransformerCap(
        num_vocab=args.num_vocab,
        d_word=args.d_word_vec,
        d_model=args.d_model,
        num_layer=args.num_layer,
        num_head=args.num_head,
        d_ff=args.d_ff,
        dropout=args.dropout)

    if args.cuda:
        model.cuda()

    #==================== Loading checkpoint ========================#

    checkpoint_path = args.checkpoint_dir+'latest.ckpt'
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        current_epoch = checkpoint['epoch']
    else:
        current_epoch = 0

    #===================== criterion and optimizer ===================#

    criterion = get_criterion(args.num_vocab)
    optimizer = optim.Adam(model.parameters(), 1e-4)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=.1)

    if os.path.isfile(checkpoint_path) and current_epoch > 0:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # optimizer = optim.Adam(model.parameters(), 1e-6)

    if current_epoch + 1 >= args.rl_finetune_epoch or args.rl_criterion_flag:
        args.batch_size = args.rl_batch_size

    #================== Loading Data ======================#

    train_dataset = CocoDataset(args, 'train')
    val_dataset = CocoDataset(args, 'test')

    vocab = train_dataset.ix_to_word

    train_data = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn)

    val_data = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn)


    #==================== Training ========================#

    # optimizer = optim.SGD(model.__get_nonconv_param__(), lr=1e-7)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=.1)
    # img_optimizer = optim.SGD(model.__get_conv_param__(), lr=1e-7)
    # img_scheduler = lr_scheduler.StepLR(img_optimizer, step_size=5, gamma=0.1)

    train(model, train_data, val_data, criterion, optimizer, vocab,  current_epoch, args)


if __name__ == "__main__":

    opt = opts.parse_opt()
    DEVICE_ID = opt.device_id
    os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_ID
    main(opt)

