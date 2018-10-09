import pickle as pkimport jsonimport argparsefrom torch.utils.data import Dataset,DataLoaderfrom torchvision import transformsfrom PIL import Imageimport torchimport osimport randomclass Vocabulary(object):    def __init__(self):        self.word2id = {}        self.id2word = {}        self.id = 0    def add_word(self, word):        if word not in self.word2id:            self.word2id[word] = self.id            self.id2word[self.id] = word            self.id += 1    def __call__(self, word):        if word not in self.word2id:            return self.word2id["<UNK>"]        return self.word2id[word]    def __len__(self):        return len(self.word2id)class CocoDataset(Dataset):    def __init__(self, root, json_path, vocab_path, max_length=18, phase='train', transform=None):        self.root = root        assert phase in ['train', 'val', 'test']        self.phase = phase        json_path = os.path.join(root, json_path)        with open(json_path, 'r') as f:            self.json = json.load(f)        vocab_path = os.path.join(root, vocab_path)        with open(vocab_path, 'rb') as f:            self.vocab = pk.load(f)        self.max_length = max_length        self.process_json()        if transform is None:            if self.phase == 'train':                self.transforms = transforms.Compose([                    transforms.Resize((224,224)),                    transforms.RandomHorizontalFlip(),                    transforms.ColorJitter(32./255., 0.5, 0.5, 0.032),                    transforms.ToTensor(),                    transforms.Normalize(mean=[0.485, 0.456, 0.406],                                         std=[0.229, 0.224, 0.225])                ])            else:                self.transforms = transforms.Compose([                    transforms.Resize((224,224)),                    transforms.ToTensor(),                    transforms.Normalize(mean=[0.485, 0.456, 0.406],                                         std=[0.229, 0.224, 0.225])                ])        else:            self.transforms = transform    def process_json(self):        imgs = {}        ann2img = {}        anns = []        i = 0        for img in self.json:            imgs[img['id']] = os.path.join(self.root, img['file_path'])            if self.phase == 'train' or self.phase == 'val':                for ann in img['processed_tokens']:                    anns.append(ann)                    ann2img[i] = img['id']                    i += 1            else:                num_anns = len(img['processed_tokens'])                ann_idx = random.randint(0, num_anns-1)                anns.append(img['processed_tokens'][ann_idx])                ann2img[i] = img['id']                i += 1        self.imgs = imgs        self.ann2img = ann2img        self.anns = anns    def __getitem__(self, idx):        img_id = self.ann2img[idx]        ann = self.anns[idx]        img_path = self.imgs[img_id]        img = Image.open(img_path).convert('RGB')        img = self.transforms(img)        target = ann + ['</S>']        if self.phase != 'test':            ann = ['<S>'] + ann            caption = torch.zeros(self.max_length+1).long()            for i, word in enumerate(ann):                if i > self.max_length:                    break                caption[i] = self.vocab(word)        else:            ann = ['<S>']            caption = torch.tensor(self.vocab(ann[0])).long()        truth = torch.zeros(self.max_length+1).long()        for i, word in enumerate(target):            if i > self.max_length:                break            truth[i] = self.vocab(word)        return img, caption, truth, img_id    def __len__(self):        return len(self.anns)def collate_fn(batch):    imgs = []    captions = []    targets = []    img_ids = []    for sample in batch:        imgs.append(sample[0])        captions.append(sample[1])        targets.append(sample[2])        img_ids.append(sample[3])    imgs = torch.stack(imgs, dim=0)    captions = torch.stack(captions, dim=0)    targets = torch.stack(targets, dim=0)    img_ids = torch.tensor(img_ids).long()    return imgs, captions, targets, img_idsdef main():    dataset = CocoDataset('./data/', 'coco_train.json', 'vocab.pkl')    data_loader = DataLoader(dataset=dataset,                             batch_size=12,                             shuffle=True,                             num_workers=6,                             collate_fn=collate_fn)    # for batch in data_loader:    #     imgs, captions, targets, img_ids = batch    #     print(imgs.size())    #     print(captions.size())    #     print(targets.size())    #     print(img_ids)    with open('./data/vocab.pkl','rb') as f:        vocab = pk.load(f)        print(len(vocab))if __name__ == '__main__':    main()