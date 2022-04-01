import json
import jieba
import torch

from models.models import BiLSTM_Attention

class Inference():
    def __init__(self,config):
        self.config = config
        self.device = torch.device("cuda:%d"%self.config.cuda if torch.cuda.is_available() else "cpu")
        with open(self.config.schema_fn,"r",encoding="utf-8") as f:
            self.id2label = json.load(f)[1]
        if self.config.flag == "char":
            with open(self.config.char_vocab,"r",encoding="utf-8") as f:
                self.vocab = json.load(f)
        else:
            with open(self.config.word_vocab,"r",encoding="utf-8") as f:
                self.vocab = json.load(f)
        self.model = BiLSTM_Attention(self.config)
        self.model.load_state_dict(torch.load(self.config.save_model,map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def __data_processing(self,sentence):
        token2id = []
        if self.config.flag == "char":
            for char in list(sentence):
                if char in self.vocab:
                    token2id.append(self.vocab[char])
                else:
                    token2id.append(self.vocab["unk"])
        else:
            for word in jieba.lcut(sentence):
                if word in self.vocab:
                    token2id.append(self.vocab[word])
                else:
                    token2id.append(self.vocab["unk"])
        input_ids = torch.LongTensor([token2id])
        return {"input_ids":input_ids}

    def predict(self,sentence):
        data = self.__data_processing(sentence)
        pred = self.model(data).cpu()
        pred_idx = pred.argmax(dim=-1)
        predict = self.id2label[str(pred_idx.item())]
        print("predict:",predict)
