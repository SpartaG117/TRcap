import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device_id', type=str, default='3,7')

    parser.add_argument('--aug_json', type=str, default='./data/generator_b_iters_21501.json')

    parser.add_argument('--aug_seq_per_img', type=int, default=5)
    parser.add_argument('--seq_per_img', type=int, default=5)

    parser.add_argument('--current_lr', type=float, default=1e-5)

    # Data input settings
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--rl_batch_size', type=int, default=10)



    parser.add_argument('--max_length', type=int, default=16)
    parser.add_argument('--rl_finetune_epoch', type=int, default=20)
    parser.add_argument('--rl_criterion_flag', type=bool, default=False)
    parser.add_argument('--cider_weight', type=float, default=0.8)


    parser.add_argument('--num_head', type=int, default=8)
    parser.add_argument('--num_layer', type=int, default=6)
    parser.add_argument('--num_vocab', type=int, default=9490)
    parser.add_argument('--d_word_vec', type=int, default=512)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=1024)


    parser.add_argument('--n_warmup_steps', type=int, default=4000)

    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embs_share_weight', action='store_true')
    parser.add_argument('--proj_share_weight', action='store_true')

    parser.add_argument('--log_path', type=str, default='./log/layer8/')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/layer8/')
    parser.add_argument('--cuda', type=bool, default=True)

    parser.add_argument('--input_adj_dir', type=str, default='./data/attr_adj')
    parser.add_argument('--input_act_dir', type=str, default='./data/attr_act')
    parser.add_argument('--input_nt_dir', type=str, default='./data/attr_nt')
    parser.add_argument('--input_rare_dir', type=str, default='./data/attr_rare')
    parser.add_argument('--input_topic_dir', type=str, default='./data/attr_topic')



    # original
    parser.add_argument('--input_json', type=str, default='./data/cocotalk.json',
                    help='path to the json file containing additional info and vocab')
    parser.add_argument('--input_fc_dir', type=str, default='./data/cocobu_fc',
                    help='path to the directory containing the preprocessed fc feats')
    parser.add_argument('--input_att_dir', type=str, default='./data/cocobu_att',
                    help='path to the directory containing the preprocessed att feats')
    parser.add_argument('--input_box_dir', type=str, default='./data/cocobu_box',
                    help='path to the directory containing the boxes of att feats')
    parser.add_argument('--input_label_h5', type=str, default='./data/cocotalk_label.h5',
                    help='path to the h5file containing the preprocessed dataset')
    parser.add_argument('--cached_tokens', type=str, default='coco-train-idxs',
                        help='Cached token file for calculating cider score during self critical training.')



    # feature manipulation
    parser.add_argument('--norm_att_feat', type=int, default=0,
                    help='If normalize attention features')
    parser.add_argument('--use_box', type=int, default=0,
                    help='If use box features')
    parser.add_argument('--norm_box_feat', type=int, default=0,
                    help='If use box, do we normalize box feature')

    #Optimization: for the Language Model
    parser.add_argument('--optim', type=str, default='adam',
                    help='what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                    help='learning rate')
    parser.add_argument('--learning_rate_decay_start', type=int, default=-1, 
                    help='at what iteration to start decaying learning rate? (-1 = dont) (in epoch)')
    parser.add_argument('--learning_rate_decay_every', type=int, default=3, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--learning_rate_decay_rate', type=float, default=0.8, 
                    help='every how many iterations thereafter to drop LR?(in epoch)')
    parser.add_argument('--optim_alpha', type=float, default=0.9,
                    help='alpha for adam')
    parser.add_argument('--optim_beta', type=float, default=0.999,
                    help='beta used for adam')
    parser.add_argument('--optim_epsilon', type=float, default=1e-8,
                    help='epsilon that goes into denominator for smoothing')
    parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight_decay')

    parser.add_argument('--scheduled_sampling_start', type=int, default=-1, 
                    help='at what iteration to start decay gt probability')
    parser.add_argument('--scheduled_sampling_increase_every', type=int, default=5, 
                    help='every how many iterations thereafter to gt probability')
    parser.add_argument('--scheduled_sampling_increase_prob', type=float, default=0.05, 
                    help='How much to update the prob')
    parser.add_argument('--scheduled_sampling_max_prob', type=float, default=0.25, 
                    help='Maximum scheduled sampling prob.')


    # Evaluation/Checkpointing
    parser.add_argument('--val_images_use', type=int, default=3200,
                    help='how many images to use when periodically evaluating the validation loss? (-1 = all)')
    parser.add_argument('--save_checkpoint_every', type=int, default=1,
                    help='how often to save a model checkpoint (in iterations)?')
    parser.add_argument('--checkpoint_path', type=str, default='save',
                    help='directory to store checkpointed models')
    parser.add_argument('--language_eval', type=int, default=1,
                    help='Evaluate language as well (1 = yes, 0 = no)? BLEU/CIDEr/METEOR/ROUGE_L? requires coco-transformer_aug code from Github.')
    parser.add_argument('--losses_log_every', type=int, default=25,
                    help='How often do we snapshot losses, for inclusion in the progress dump? (0 = disable)')       
    parser.add_argument('--load_best_score', type=int, default=1,
                    help='Do we load previous best score when resuming training.')       

    # misc
    parser.add_argument('--id', type=str, default='',
                    help='an id identifying this run/job. used in cross-val and appended when writing progress files')
    parser.add_argument('--train_only', type=int, default=0,
                    help='if true then use 80k, else use 110k')


    # Reward
    parser.add_argument('--cider_reward_weight', type=float, default=1,
                    help='The reward weight from cider')
    parser.add_argument('--bleu_reward_weight', type=float, default=0,
                    help='The reward weight from bleu4')

    args = parser.parse_args()


    return args
