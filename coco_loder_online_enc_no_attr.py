import pickle as pk
import json
import argparse
from torch.utils.data import Dataset, DataLoader
import torch
import os
import random
import h5py
import numpy as np
import opts

class CocoDataset(Dataset):

    def __init__(self, opt, mode='test'):
        self.opt = opt
        self.mode = mode
        self.batch_size = self.opt.batch_size

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        # add start token, end token and pad token into dictionary
        self.ix_to_word['0'] = '<pad>'
        self.ix_to_word['9488'] = '<s>'
        self.ix_to_word['9489'] = '</s>'
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)

        data_dir = 'data/' + mode +'_online_bu/'
        self.input_fc_dir = data_dir + 'fc'
        self.input_att_dir = data_dir + 'att'
        self.input_box_dir = data_dir + 'box'

        self.image_ids = []
        file_list = os.listdir(self.input_fc_dir)
        for file_name in file_list:
            img_id = file_name.split('.')[0]
            self.image_ids.append(img_id)
        print('load', mode)

    def get_visual_feature(self, img_id):
        att_feat = np.load(os.path.join(self.input_att_dir, img_id + '.npz'))['feat']
        fc_feat = np.load(os.path.join(self.input_fc_dir, img_id + '.npy'))

        att_feat = torch.from_numpy(att_feat)
        fc_feat = torch.from_numpy(fc_feat)

        return att_feat, fc_feat

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, ix):
        img_id = self.image_ids[ix]
        att_feat, fc_feat = self.get_visual_feature(img_id)

        return att_feat, fc_feat, img_id


def collate_fn(batch):
    att_batch = []
    fc_batch = []
    id_batch = []

    for sample in batch:
        att_batch.append(sample[0])
        fc_batch.append(sample[1])
        id_batch.append(sample[2])

    max_att_len = max([_.size(0) for _ in att_batch])
    att_feats = torch.zeros([len(att_batch), max_att_len, att_batch[0].size(-1)]).float()
    for i in range(len(att_batch)):
        att_feats[i, :att_batch[i].size(0), :] = att_batch[i]

    fc_batch = torch.stack(fc_batch, dim=0)

    return att_feats, fc_batch, id_batch



if __name__ == '__main__':
    opt = opts.parse_opt()
    train_dataset = CocoDataset(opt, 'test')
    data_loader = DataLoader(train_dataset,
                             batch_size=128,
                             shuffle=True,
                             num_workers=4,
                             collate_fn=collate_fn)
    print(len(train_dataset))
    # print(train_dataset[0][0].shape)
    # print(train_dataset[0][1].shape)
    # print(train_dataset[0][2].shape)
    # print(train_dataset[0][3].shape)
    # print(train_dataset[0][4])

    for idx, data in enumerate(data_loader):
        print(data[0].shape)
        print(data[1].shape)
        print(data[2])

