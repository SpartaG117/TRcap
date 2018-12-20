import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from Model.Model import TransformerCap
from coco_loader_h5 import CocoDataset, collate_fn
import json
import os
import opts
from beamsearch_batch import Beamsearch

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
DEVICE_ID = '10'
os.environ['CUDA_VISIBLE_DEVICES'] = DEVICE_ID


test_dataset = CocoDataset(opt, 'test')
vocab = test_dataset.ix_to_word
test_data = DataLoader(
    test_dataset,
    batch_size=168,
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

check_point = torch.load('checkpoints/d_ff_1024/epoch_24_score_1.130.ckpt')
model.load_state_dict(check_point['model'])
model.cuda()
model.eval()


results = []
with torch.no_grad():
    for batch in tqdm(test_data, total=len(test_data)):

        _, _, att_feat, fc_feat, infos = batch

        att_feat = att_feat.unsqueeze(1)
        fc_feat = fc_feat.unsqueeze(1)

        batch_size = att_feat.size(0)
        beam_size = 3

        searcher = Beamsearch(beam_size=beam_size, batch_size=batch_size, max_len=18)

        att_feat = att_feat.expand(-1, beam_size, -1, -1).contiguous().view(batch_size*beam_size, att_feat.size(2), -1).cuda()
        fc_feat = fc_feat.expand(-1, beam_size, -1).contiguous().view(batch_size*beam_size, -1).cuda()

        cap = searcher.get_input_cap()
        cap = torch.Tensor(cap).long()
        cap = cap.cuda()
        pred = model(cap, att_feat, fc_feat, return_attn=False)

        while searcher.search(pred):
            cap = searcher.get_input_cap()
            cap = torch.Tensor(cap).long()
            cap = cap.cuda()
            pred = model(cap, att_feat, fc_feat, return_attn=False)

        pred_cap = searcher.get_results()

        for b in range(batch_size):
            cap_word = ''
            for word in pred_cap[b]:
                cap_word += vocab[str(word)]
                cap_word += ' '
            if cap_word[-1] == ' ':
                cap_word = cap_word[:-1]

            caption = {}
            caption['image_id'] = infos[b]['image_id']
            caption['caption'] = cap_word
            results.append(caption)

# json.dump(results, open('./eval/results/beam4_best.json', 'w'))
print(language_eval(results, 'd_ff_1024_24_beam_3', 'test'))

