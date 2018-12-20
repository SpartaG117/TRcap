from train_utils import trans2sentence
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Model_enc_no_attr_fc.Model import TransformerCap
from Model_enc_no_attr.Model import TransformerCap as TransformerCap_pos
from Model_enc_no_attr_fc_1.Model import TransformerCap as TransformerCap_1

from coco_loader_h5 import CocoDataset, collate_fn
import json
import os
import opts
from ensemble_model import EnsembleNoAttributeProb



opt = opts.parse_opt()
DEVICE_ID = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_ID


test_dataset = CocoDataset(opt, 'test')
# val_dataset = CocoDataset(opt, 'val')

test_data = DataLoader(
    test_dataset,
    batch_size=200,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn)

# val_data = DataLoader(
#     val_dataset,
#     batch_size=128,
#     shuffle=False,
#     num_workers=4,
#     collate_fn=collate_fn)


model1 = TransformerCap_1(
    num_vocab=9490,
    d_word=512,
    d_model=512,
    num_layer=6,
    num_head=8,
    d_ff=1024,
    dropout=0.1)

model2 = TransformerCap_1(
    num_vocab=9490,
    d_word=512,
    d_model=512,
    num_layer=6,
    num_head=8,
    d_ff=1024,
    dropout=0.1)

model3 = TransformerCap_1(
    num_vocab=9490,
    d_word=512,
    d_model=512,
    num_layer=6,
    num_head=8,
    d_ff=1024,
    dropout=0.1)

model4 = TransformerCap(
    num_vocab=9490,
    d_word=512,
    d_model=512,
    num_layer=6,
    num_head=8,
    d_ff=1024,
    dropout=0.1)

model5 = TransformerCap(
    num_vocab=9490,
    d_word=512,
    d_model=512,
    num_layer=6,
    num_head=8,
    d_ff=1024,
    dropout=0.1)

model6 = TransformerCap(
    num_vocab=9490,
    d_word=512,
    d_model=512,
    num_layer=6,
    num_head=8,
    d_ff=1024,
    dropout=0.1)

model_l8 = TransformerCap_1(
    num_vocab=9490,
    d_word=512,
    d_model=512,
    num_layer=8,
    num_head=8,
    d_ff=1024,
    dropout=0.1)

model_pos = TransformerCap_pos(
    num_vocab=9490,
    d_word=512,
    d_model=512,
    num_layer=6,
    num_head=8,
    d_ff=1024,
    n_max_seq=16,
    dropout=0.1)

check_point1 = torch.load('checkpoints/ensemble/1211/epoch_303_cider_1.275_bleu4_0.389.ckpt')
check_point2 = torch.load('checkpoints/ensemble/1211/epoch_466_cider_1.2734_bleu4_0.3881.ckpt')
check_point3 = torch.load('checkpoints/ensemble/1211/epoch_431_cider_1.266_bleu4_0.388.ckpt')
check_point4 = torch.load('checkpoints/ensemble/1211/epoch_643_cider_1.267_bleu4_0.386.ckpt')
check_point5 = torch.load('checkpoints/ensemble/1211/epoch_500_cider_1.267_bleu4_0.392.ckpt')
check_point_pos = torch.load('checkpoints/ensemble/1212/epoch_609_cider_1.2704_bleu4_0.3890.ckpt')
check_point_l8 = torch.load('checkpoints/ensemble/1212/epoch_341_cider_1.2697_bleu4_0.3893.ckpt')


model1.load_state_dict(check_point1['model'])
model2.load_state_dict(check_point2['model'])
model3.load_state_dict(check_point3['model'])
model4.load_state_dict(check_point4['model'])
model5.load_state_dict(check_point5['model'])
model_pos.load_state_dict(check_point_pos['model'])
model_l8.load_state_dict(check_point_l8['model'])


ensemble_model = EnsembleNoAttributeProb([model1, model2, model3, model4, model5, model_pos, model_l8], 16)
ensemble_model = ensemble_model.cuda()


res = []
with torch.no_grad():
    for idx, batch in tqdm(enumerate(test_data), mininterval=2,
                           desc='  -(evalution test)  ', leave=False, total=len(test_data)):

        _, _, att_feat, fc_feat, infos = batch

        att_feat = att_feat.squeeze(1).cuda()
        fc_feat = fc_feat.squeeze(1).cuda()

        # Tensor of shape [B x 2048 x 7 x 7]
        preds = ensemble_model(att_feat, fc_feat)
        pred_cap = trans2sentence(preds, test_dataset.ix_to_word)

        for i, info in enumerate(infos):
            cap_dict = {}
            cap_dict['image_id'] = info['image_id']
            cap_dict['caption'] = pred_cap[i]
            res.append(cap_dict)

    # for idx, batch in tqdm(enumerate(val_data), mininterval=2,
    #                        desc='  -(evalution val)  ', leave=False, total=len(test_data)):
    #
    #     _, _, att_feat, fc_feat, infos = batch
    #
    #     att_feat = att_feat.squeeze(1).cuda()
    #     fc_feat = fc_feat.squeeze(1).cuda()
    #
    #     # Tensor of shape [B x 2048 x 7 x 7]
    #     preds = ensemble_model(att_feat, fc_feat)
    #     pred_cap = trans2sentence(preds, test_dataset.ix_to_word)
    #
    #     for i, info in enumerate(infos):
    #         cap_dict = {}
    #         cap_dict['image_id'] = info['image_id']
    #         cap_dict['caption'] = pred_cap[i]
    #         res.append(cap_dict)


path = 'gen/1212/ensemble8.json'
with open(path, 'w') as f:
    json.dump(res, f)




