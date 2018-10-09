import osimport jsonimport argparsefrom random import shuffle, seedimport stringfrom collections import Counterimport pickle as pkclass Vocabulary(object):    def __init__(self):        self.word2id = {}        self.id2word = {}        self.id = 0    def add_word(self, word):        if word not in self.word2id:            self.word2id[word] = self.id            self.id2word[self.id] = word            self.id += 1    def __call__(self, word):        if word not in self.word2id:            return self.word2id["<UNK>"]        return self.word2id[word]    def __len__(self):        return len(self.word2id)def prepro_captions(imgs):    print('example processed tokens:')    for img in imgs:        img['processed_tokens'] = []        for s in img['captions']:            s = str.encode(s.lower())            text = s.translate(None, str.encode(string.punctuation)).decode().strip().split()            img['processed_tokens'].append(text)def build_vocab(imgs,params):    counter = Counter()    max_length = params['max_length']    for img in imgs:        for s in img['processed_tokens']:            counter.update(s[0:max_length])    words = [word for word,count in counter.items() if count >= params['word_count_threshold']]    bad_words = [word for word,count in counter.items() if count < params['word_count_threshold']]    print("total words",len(words)+len(bad_words))    print('good words',len(words))    print("number of bad words: %d/%d = %.2f%%" %(len(bad_words),len(bad_words)+len(words),                                         len(bad_words)*100.0/(len(bad_words)+len(words)) ))    # build vocabulary for training captions    # add some special tokens    vocab = Vocabulary()    vocab.add_word('<pad>')    vocab.add_word('<S>')    vocab.add_word('</S>')    vocab.add_word('<UNK>')    for word in words:        vocab.add_word(word)    print('write vocab counts doc')    counter = sorted(counter.items(), key = lambda x:x[1], reverse=True)    with open(params['vocab_counter_path'],'w') as f:        for word,cnt in counter:            line = word + ':' + str(cnt) +'\n'            f.write(line)    return vocabdef assign_splits(imgs, params):    num_val = params['num_val']    num_test = params['num_test']    for i,img in enumerate(imgs):      if i < num_val:        img['split'] = 'val'      elif i < num_val + num_test:        img['split'] = 'test'      else:        img['split'] = 'train'    print ('assigned %d to val, %d to test.' % (num_val, num_test))def main(params):    with open(params['input_json'],'rb') as f:        imgs = json.load(f)    seed(123)    shuffle(imgs)    prepro_captions(imgs)    train = assign_splits(imgs,params)    print('write vocab.pkl')    vocab = build_vocab(imgs[10000:],params)    with open(params['vocab_path'],'wb') as f:        pk.dump(vocab,f)    print('write output json')    with open(params['output_json'],'w') as f:        json.dump(imgs,f)if __name__ == "__main__":    parser = argparse.ArgumentParser()    # input json    parser.add_argument('--input_json', default='./data/coco_raw.json',                        help='input json file to process into hdf5')    parser.add_argument('--output_json', default='./data/coco_processed.json', help='output json file')    parser.add_argument('--vocab_counter_path', type = str,                        default = './data/vocab_counter.txt')    parser.add_argument('--vocab_path', type = str,                        default = './data/vocab.pkl',                        help = 'path for vocabulary wrapper')    parser.add_argument('--max_length', default=18, type=int,                         help='max length of a caption, in number of words. captions longer than this get clipped.')    # parser.add_argument('--images_root', default='',    #                     help='root location in which images are stored, to be prepended to file_path in input json')    parser.add_argument('--word_count_threshold', default=5, type=int,                        help='only words that occur more than this number of times will be put in vocab')    parser.add_argument('--num_val', default=5000, type=int,                        help='number of images to assign to validation data (for CV etc)')    parser.add_argument('--num_test', default=5000, type=int,                        help='number of test images (to withold until very very end)')    args = parser.parse_args()    params = vars(args)    print(json.dumps(params,indent=2))    main(params)