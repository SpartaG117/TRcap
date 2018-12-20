from train_h5 import search, trans2sentence
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Model.Model import TransformerCap
from coco_loader_h5 import CocoDataset, collate_fn
import json
import os
import opts

def language_eval(preds, model_id, split):

    annFile = 'coco_caption/annotations/captions_val2014.json'
    from coco_caption.pycocotools.coco import COCO
    from coco_caption.pycocoevalcap.eval import COCOEvalCap

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if p['image_id'] in valids]
    print(len(preds_filt))
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out


opt = opts.parse_opt()
DEVICE_ID = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_ID


test_dataset = CocoDataset(opt, 'test')

test_data = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn)

model = TransformerCap(
    num_vocab=9490,
    d_word=512,
    d_model=512,
    num_layer=6,
    num_head=8,
    d_ff=1024,
    dropout=0.1)

check_point = torch.load('checkpoints/d_ff_1024/epoch_14_score_1.069.ckpt')
model.load_state_dict(check_point['model'])
model.cuda()
model.eval()
res = []

with torch.no_grad():
    for idx, batch in tqdm(enumerate(test_data), mininterval=2,
                           desc='  -(evalution)  ', leave=False, total=len(test_data)):

        _, _, att_feat, fc_feat, infos = batch

        att_feat = att_feat.squeeze(1).cuda()
        fc_feat = fc_feat.squeeze(1).cuda()

        # Tensor of shape [B x 2048 x 7 x 7]
        preds = search(att_feat, fc_feat, model, 18)
        pred_cap = trans2sentence(preds, test_dataset.ix_to_word)

        for i, info in enumerate(infos):
            cap_dict = {}
            cap_dict['image_id'] = info['image_id']
            cap_dict['caption'] = pred_cap[i]
            res.append(cap_dict)


print(language_eval(res, 'dff_1024_14', 'test'))




