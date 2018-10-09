import torch
from Model1 import TransformerCap
from torch.utils.data import DataLoader
from coco_loader import CocoDataset,collate_fn,Vocabulary
from beamsearch import Beamsearch
import pickle as pk
from tqdm import tqdm
import json

val_dataset = CocoDataset(
    root='./data/',
    json_path='coco_test.json',
    vocab_path='vocab.pkl',
    max_length=18,
    phase='test')

val_data = DataLoader(
    val_dataset,
    batch_size=100,
    shuffle=False,
    num_workers=8,
    collate_fn=collate_fn)

model = TransformerCap(
    num_vocab=10054,
    d_word=512,
    d_model=512,
    num_layer=4,
    num_head=8,
    d_ff=1024,
    dropout=0.1)

checkpoint = torch.load('./checkpoints/epoch_22_acc_52.238.ckpt')
model.load_state_dict(checkpoint['model'])
with open('./data/vocab.pkl', 'rb') as f:
    vocab = pk.load(f)

for param in model.parameters():
    param.requires_grad = False
model.cuda()
model.eval()
# for batch in val_data:
#     img, cap, target, img_id = batch
#     #img = img.cuda()
#     cap = cap.unsqueeze(0)
#     print(cap.size())
#     beam_search = Beamsearch(beam_size=3, max_len=18)
#     pred = model(cap, img)
#     flag = 1
#     while(beam_search.search(pred)):
#         beams = []
#         for beam in beam_search.beams:
#             beams.append(beam['word'])
#         beams = torch.tensor(beams).long()
#         pred =  model(beams, img)
#         flag += 1
#     print(flag)
#     caption = {}
#     caption['image_id'] = img_id.items()
#     caption['caption'] = beam_search.get_results()
#     print(caption)
#     break
results = []
for batch in tqdm(val_data, total=len(val_data)):
    img, cap, target, img_id = batch
    img = img.cuda()  # b x 3 x 224 x 224
    cap = cap.unsqueeze(1)  # b x 1

    cap = cap.cuda()
    while cap.size(1) <= 19:
        pred = model(cap, img)
        pred = pred.view(cap.size(0), cap.size(1), -1)
        pred = pred[:,-1,:]  # b x num_word
        cap = torch.cat([cap, pred.max(1)[1].unsqueeze(1)], dim=1)

    cap_id = cap[:,:].tolist()
    for i in range(len(cap_id)):
        cap_word = ''
        for word in cap_id[i]:
            if word == 1:
                continue
            elif word == 2:
                cap_word = cap_word[:-1]
                break
            cap_word += vocab.id2word[word]
            cap_word += ' '
        if cap_word[-1] == ' ':
            cap_word = cap_word[:-1]
        caption = {}
        caption['image_id'] = img_id[i].item()
        caption['caption'] = cap_word
        results.append(caption)
json.dump(results, open('./eval/results/epoch_22_acc_52.238.json', 'w'))


