import datetime
import pandas as pd
import torch
from models.models import BiLSTM_Attention
import json
from dataloader.dataloader import MyDataset,collate_fn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from logger.logger import Logger
from tqdm import tqdm

class Framework():
    def __init__(self,config):
        self.config = config
        self.device = torch.device("cuda:%d"%self.config.cuda if torch.cuda.is_available() else "cpu")
        with open(self.config.schema_fn,"r",encoding="utf-8") as f:
            self.id2label = json.load(f)[1]
        self.logger = Logger(self.config.log.format(datetime.datetime.now().strftime("%Y-%m-%d - %H:%M:%S")))

    def train(self):

        train_dataset = MyDataset(self.config,self.config.train_fn)
        dev_dataset = MyDataset(self.config,self.config.dev_fn)

        train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=self.config.batch_size,
                                      pin_memory=True,collate_fn=collate_fn)
        dev_dataloader = DataLoader(dev_dataset,batch_size=self.config.batch_size,
                                    pin_memory=True,collate_fn=collate_fn)

        model = BiLSTM_Attention(self.config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=self.config.learning_rate)
        loss_fn = torch.nn.NLLLoss()

        best_epoch = 0
        best_f1_score,precision,recall = 0,0,0
        global_step = 0
        global_loss = 0
        accuracy = 0
        for epoch in range(1,self.config.epoch+1):
            print("Epoch [{}/{}]".format(epoch,self.config.epoch))
            for data in tqdm(train_dataloader):
                pred = model(data)
                optimizer.zero_grad()
                loss = loss_fn(pred,data["target"].to(self.device))
                loss.backward()
                optimizer.step()
                global_loss += loss.item()
                if (global_step+1) % self.config.step == 0:
                    self.logger.logger.info("epoch:{} global_step:{} global_loss:{:5.4f}".
                                            format(epoch,global_step,global_loss))
                    global_loss = 0
                global_step += 1
            r, p, f1_score, predict, sentence, gold, acc = self.evaluate(model,dev_dataloader)
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                recall = r
                precision = p
                best_epoch = epoch
                if acc > accuracy:
                    accuracy = acc
                pd.DataFrame({"sentence":sentence,"label":gold,"predict":predict}).to_csv(self.config.dev_result)
                print("epoch:{} save model......".format(epoch))
                torch.save(model.state_dict(),self.config.save_model)
            self.logger.logger.info("epoch:{} recall:{:5.4f} precision:{:5.4f} f1_score:{:5.4f} best_f1_score:{:5.4f} accuracy:{:5.4f} best_epoch:{}".
                                    format(epoch,recall,precision,f1_score,best_f1_score,accuracy,best_epoch))

    def evaluate(self,model,dataloader):

        model.eval()
        predict_num,gold_num,correct_num=0,0,0
        correct = 0
        gold = []
        predict = []
        sentence = []
        with torch.no_grad():
            for data in tqdm(dataloader):
                pred = model(data)
                pred = F.softmax(pred)
                pred_id = pred.argmax(dim=-1)
                label = data["label"]
                sentence.extend(data["sentence"])
                resl = []
                for idx in pred_id:
                    resl.append(self.id2label[str(idx.cpu().item())])
                for k,v in enumerate(label):
                    if v == resl[k] and v != self.config.label_flag:
                        correct_num += 1
                    if v != self.config.label_flag:
                        gold_num += 1
                for k,v in enumerate(resl):
                    if v != self.config.label_flag:
                        predict_num += 1
                for k,v in enumerate(label):
                    if v == resl[k]:
                        correct += 1
                gold.extend(label)
                predict.extend(resl)
        recall = correct_num / (gold_num + 1e-10)
        precision = correct_num / (predict_num + 1e-10)
        f1_score = 2 * recall * precision / (recall + precision)
        accuracy = correct / (len(sentence) + 1e-10)
        print("predict_num:{} gold_num:{} correct_num:{}".format(predict_num,gold_num,correct_num))
        model.train()
        return recall,precision,f1_score,predict,sentence,gold,accuracy

    def test(self):

        model = BiLSTM_Attention(self.config)
        model.load_state_dict(torch.load(self.config.save_model,map_location=self.device))
        model.eval()
        model.to(self.device)

        dataset = MyDataset(self.config,self.config.test_fn)
        dataloader = DataLoader(dataset,shuffle=True,batch_size=self.config.batch_size,
                                collate_fn=collate_fn,pin_memory=True)
        predict_num,gold_num,correct_num=0,0,0
        correct = 0
        gold = []
        predict = []
        sentence = []
        with torch.no_grad():
            for data in tqdm(dataloader):
                pred = model(data)
                pred = F.softmax(pred)
                pred_id = pred.argmax(dim=-1)
                label = data["label"]
                sentence.extend(data["sentence"])
                resl = []
                for idx in pred_id:
                    resl.append(self.id2label[str(idx.cpu().item())])
                for k,v in enumerate(label):
                    if v == resl[k] and v != self.config.label_flag:
                        correct_num += 1
                    if v != self.config.label_flag:
                        gold_num += 1
                for k,v in enumerate(resl):
                    if v != self.config.label_flag:
                        predict_num += 1
                for k,v in enumerate(label):
                    if v == resl[k]:
                        correct += 1
                gold.extend(label)
                predict.extend(resl)
        recall = correct_num / (gold_num + 1e-10)
        precision = correct_num / (predict_num + 1e-10)
        f1_score = 2 * recall * precision / (recall + precision)
        accuracy = correct / (len(sentence) + 1e-10)
        print("predict_num:{} gold_num:{} correct_num:{}".format(predict_num,gold_num,correct_num))
        model.train()
        return recall,precision,f1_score,predict,sentence,gold,accuracy
