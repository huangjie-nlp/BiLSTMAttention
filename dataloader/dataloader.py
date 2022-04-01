import torch
from torch.utils.data import Dataset
import pandas as pd
import json
import numpy as np
import jieba

class MyDataset(Dataset):
    def __init__(self,config,fn):
        self.config = config
        self.df = pd.read_csv(fn)
        self.sentence = self.df.sentence.tolist()
        self.label = self.df.label.tolist()
        with open(self.config.schema_fn,"r",encoding="utf-8") as f:
            self.label2id = json.load(f)[0]
        if self.config.flag == "char":
            with open(self.config.char_vocab,"r",encoding="utf-8") as f:
                self.vocab = json.load(f)
        elif self.config.flag == 'word':
            with open(self.config.word_vocab,"r",encoding="utf-8") as f:
                self.vocab = json.load(f)

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, idx):
        label = self.label[idx]
        sentence = self.sentence[idx]
        if self.config.flag == "char":
            token = list(sentence)
            input_ids = []
            for char in token:
                if char in self.vocab:
                    input_ids.append(self.vocab[char])
                else:
                    input_ids.append(self.vocab["unk"])
            token_len = len(token)
        else:
            token = jieba.lcut(sentence)
            input_ids = []
            for word in token:
                if word in self.vocab:
                    input_ids.append(self.vocab[word])
                else:
                    input_ids.append(self.vocab["unk"])
            token_len = len(token)
        label2id = self.label2id[label]
        mask = [1] * token_len
        input_ids = np.array(input_ids)
        mask = np.array(mask)
        token_len = len(input_ids)

        return sentence,label,token_len,input_ids,mask,label2id

def collate_fn(batch):
    sentence, label, token_len, input_ids, mask, label2id = zip(*batch)
    cur_batch = len(batch)
    max_len = max(token_len)

    batch_input_ids = torch.LongTensor(cur_batch,max_len).zero_()
    batch_mask = torch.LongTensor(cur_batch,max_len).zero_()

    for i in range(cur_batch):
        batch_input_ids[i,:token_len[i]].copy_(torch.from_numpy(input_ids[i]))
        batch_mask[i,:token_len[i]].copy_(torch.from_numpy(mask[i]))

    return {"sentence":sentence,
            "label":label,
            "input_ids":batch_input_ids,
            "mask":batch_mask,
            "target":torch.LongTensor(label2id)}
