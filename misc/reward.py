import torch
from train_utils import calc_bleu

def get_self_critical_reward(base_sents, sample_sents, gts, batch_size, max_seq_len, cider):

    sample_res = {}
    baseline_res = {}
    for i in range(batch_size):
        sample_res[i] = [sample_sents[i]]
        baseline_res[i] = [base_sents[i]]

    _, s_rewards = cider.compute_score(gts, sample_res)
    _, b_rewards = cider.compute_score(gts, baseline_res)

    rewards = s_rewards - b_rewards
    avg_rewards = rewards.mean()

    rewards = torch.Tensor(rewards).float().unsqueeze(1).expand(-1, max_seq_len)
    rewards = rewards.contiguous().view(-1).cuda()

    return rewards, avg_rewards


def get_self_critical_reward_with_bleu(base_sents, sample_sents, gts, batch_size, max_seq_len, cider, opt):

    sample_res = {}
    baseline_res = {}
    for i in range(batch_size):
        sample_res[i] = [sample_sents[i]]
        baseline_res[i] = [base_sents[i]]

    _, cider_s_rewards = cider.compute_score(gts, sample_res)
    _, cider_b_rewards = cider.compute_score(gts, baseline_res)
    _, bleu_s_rewards = calc_bleu(gts, sample_res)
    _, bleu_b_rewards = calc_bleu(gts, baseline_res)

    cider_s_rewards = torch.Tensor(cider_s_rewards)
    cider_b_rewards = torch.Tensor(cider_b_rewards)
    bleu_s_rewards = torch.Tensor(bleu_s_rewards)
    bleu_b_rewards = torch.Tensor(bleu_b_rewards)

    cider_weight = opt.cider_weight
    bleu_weight = 1 - cider_weight
    s_rewards = cider_weight * cider_s_rewards + bleu_weight * bleu_s_rewards[3]
    b_rewards = cider_weight * cider_b_rewards + bleu_weight * bleu_b_rewards[3]
    rewards = s_rewards - b_rewards

    avg_rewards = rewards.mean().item()

    rewards = rewards.float().unsqueeze(1).expand(-1, max_seq_len)
    rewards = rewards.contiguous().view(-1).cuda()

    return rewards, avg_rewards
