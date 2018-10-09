import torch
from Model1 import TransformerCap
from torch.utils.data import DataLoader
from coco_loader import CocoDataset, collate_fn, Vocabulary
from beamsearch_batch import Beamsearch
import pickle as pk
from tqdm import tqdm
import time
import json

val_dataset = CocoDataset(
    root='./data/',
    json_path='coco_test.json',
    vocab_path='vocab.pkl',
    max_length=18,
    phase='test')

val_data = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=8,
    collate_fn=collate_fn)

model = TransformerCap(
    num_vocab=10054,
#   max_len=18,
    d_word=512,
    d_model=512,
    num_layer=6,
    num_head=8,
    d_ff=1024,
    dropout=0.1)

checkpoint = torch.load('./checkpoints/best.ckpt')
model.load_state_dict(checkpoint['model'])
with open('./data/vocab.pkl', 'rb') as f:
    vocab = pk.load(f)

for param in model.parameters():
    param.requires_grad = False
model.cuda()
model.eval()

results = []
for batch in tqdm(val_data, total=len(val_data)):
    img, _, _, img_id = batch

    batch_size = img.size(0)
    beam_size = 4

    searcher = Beamsearch(beam_size=beam_size, batch_size=batch_size, max_len=18)

    imgs = []
    for b in range(batch_size):
        imgs.append(img[b].expand(beam_size, 3, 224, 224))
    imgs = torch.cat(imgs, dim=0)
    imgs = imgs.cuda()  # (b x beam_size) x 3 x 224 x 224

    cap = searcher.get_input_cap()
    cap = torch.tensor(cap).long()
    cap = cap.cuda()
    pred = model(cap, imgs)

    flag = 1
    while searcher.search(pred):
        cap = searcher.get_input_cap()
        cap = torch.tensor(cap).long()
        cap = cap.cuda()
        pred = model(cap, imgs)
        flag += 1

    pred_cap = searcher.get_results()
    for b in range(batch_size):
        cap_word = ''
        for word in pred_cap[b]:
            cap_word += vocab.id2word[word]
            cap_word += ' '
        if cap_word[-1] == ' ':
            cap_word = cap_word[:-1]

        caption = {}
        caption['image_id'] = img_id[b].item()
        caption['caption'] = cap_word
        results.append(caption)

json.dump(results, open('./eval/results/beam4_best.json', 'w'))
