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

    def __init__(self, opt, mode='train'):
        self.opt = opt
        self.mode = mode
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img

        # feature related options
        self.use_att = getattr(opt, 'use_att', True)
        self.use_box = getattr(opt, 'use_box', 0)
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)

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

        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir, opt.input_box_dir, opt.input_label_h5)
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')

        self.input_fc_dir = self.opt.input_fc_dir
        self.input_att_dir = self.opt.input_att_dir
        self.input_box_dir = self.opt.input_box_dir

        # load in the sequence data
        seq_size = self.h5_label_file['labels'].shape
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        # separate out indexes for each of the provided splits
        self.split_ix = []
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == self.mode:
                self.split_ix.append(ix)
            if self.mode == 'train' and img['split'] == 'restval':
                self.split_ix.append(ix)
            if self.mode == 'train' and img['split'] == 'val':
                self.split_ix.append(ix)

        with open('data/dataset_coco.json', 'r') as f:
            raw_json = json.load(f)
            self.raw_gt = raw_json['images']

        print(('assigned %d images to split '+ self.mode) %len(self.split_ix))


    def get_captions(self, ix, seq_per_img):

        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1 # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype=np.int32)
            for q in range(seq_per_img):
                ixl = random.randint(ix1, ix2)
                seq[q, :] = self.h5_label_file['labels'][ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.h5_label_file['labels'][ixl: ixl + seq_per_img, :self.seq_length]

        # transform numpy seq into a list seq without 0 pad
        cap_list = []
        for i in range(len(seq)):
            cap_list.append(seq[i, seq[i] > 0].tolist())

        # make seq data
        src_seq = []
        tgt_seq = []
        for cap in cap_list:
            src_seq.append([9488] + cap + [0] * (self.seq_length - len(cap)))
            tgt_seq.append(cap + [9489] + [0] * (self.seq_length - len(cap)))

        src_seq = torch.Tensor(src_seq).long()
        tgt_seq = torch.Tensor(tgt_seq).long()

        gts_seq = self.h5_label_file['labels'][self.label_start_ix[ix]-1: self.label_end_ix[ix]]
        gts = []
        for i in range(len(gts_seq)):
            gts.append(gts_seq[i, gts_seq[i] > 0].tolist())

        raw_gts = []
        raw_sentences = self.raw_gt[ix]['sentences']
        for i in range(len(raw_sentences)):
            tmp = raw_sentences[i]['tokens']
            captions = ' '.join(tmp).strip()
            raw_gts.append(captions)

        return src_seq, tgt_seq, gts, raw_gts

    def get_visual_feature(self, ix):
        att_feat = np.load(os.path.join(self.input_att_dir, str(self.info['images'][ix]['id']) + '.npz'))['feat']
        fc_feat = np.load(os.path.join(self.input_fc_dir, str(self.info['images'][ix]['id']) + '.npy'))

        att_feat = torch.from_numpy(att_feat)
        fc_feat = torch.from_numpy(fc_feat)

        return att_feat, fc_feat

    def __len__(self):
        return len(self.split_ix)

    def __getitem__(self, ix):
        index = self.split_ix[ix]

        src_seq, tgt_seq, gts, raw_gts = self.get_captions(index, self.seq_per_img)

        att_feat, fc_feat = self.get_visual_feature(index)

        info = dict()
        info['gts'] = gts
        info['image_id'] = self.info['images'][index]['id']
        info['file_path'] = self.info['images'][index]['file_path']
        info['raw_gts'] = raw_gts

        return src_seq, tgt_seq,  att_feat, fc_feat, info,


def collate_fn(batch):
    src_batch = []
    tgt_batch = []
    att_batch = []
    fc_batch = []
    info_batch = []

    for sample in batch:
        src_batch.append(sample[0])
        tgt_batch.append(sample[1])
        att_batch.append(sample[2])
        fc_batch.append(sample[3])
        info_batch.append(sample[4])

    max_att_len = max([_.size(0) for _ in att_batch])
    att_feats = torch.zeros([len(att_batch), max_att_len, att_batch[0].size(-1)]).float()
    for i in range(len(att_batch)):
        att_feats[i, :att_batch[i].size(0), :] = att_batch[i]

    src_batch = torch.stack(src_batch, dim=0)
    tgt_batch = torch.stack(tgt_batch, dim=0)

    fc_batch = torch.stack(fc_batch, dim=0)


    return src_batch, tgt_batch, att_feats, fc_batch, info_batch



if __name__ == '__main__':
    opt = opts.parse_opt()
    train_dataset = CocoDataset(opt, 'test')
    data_loader = DataLoader(train_dataset,
                             batch_size=2,
                             shuffle=True,
                             num_workers=4,
                             collate_fn=collate_fn)
    # print(len(train_dataset))
    # print(train_dataset[0][0].shape)
    # print(train_dataset[0][1].shape)
    # print(train_dataset[0][2].shape)
    # print(train_dataset[0][3].shape)
    # print(train_dataset[0][4])

    for idx, data in enumerate(data_loader):
        # print(data[0].shape)
        # print(data[1].shape)
        # print(data[2].shape)
        # print(data[3].shape)

        print(data[4])

