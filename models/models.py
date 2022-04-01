
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BiLSTM_Attention(nn.Module):
    def __init__(self,config):
        super(BiLSTM_Attention, self).__init__()
        self.config = config
        self.device = torch.device("cuda:%d"%self.config.cuda if torch.cuda.is_available() else "cpu")
        if self.config.flag == "char":
            self.embedding = nn.Embedding(self.config.char_num,self.config.embed_dim)
        else:
            self.embedding = nn.Embedding(self.config.word_num,self.config.embed_dim)
        self.lstm = nn.LSTM(self.config.embed_dim,self.config.unit//2,batch_first=True,bidirectional=True,num_layers=1)
        self.fc = nn.Linear(self.config.unit,self.config.label_num)
        self.dropout = nn.Dropout(self.config.dropout)
        self.ln = nn.LayerNorm(self.config.unit)

    def attention_net(self, x, query, mask=None):

        d_k = query.size(-1)     # d_k为query的维度

        # query:[batch, seq_len, hidden_dim*2], x.t:[batch, hidden_dim*2, seq_len]
#         print("query: ", query.shape, x.transpose(1, 2).shape)  # torch.Size([128, 38, 128]) torch.Size([128, 128, 38])
        # 打分机制 scores: [batch, seq_len, seq_len]
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)
#         print("score: ", scores.shape)  # torch.Size([128, 38, 38])

        # 对最后一个维度 归一化得分
        alpha_n = F.softmax(scores, dim=-1)
#         print("alpha_n: ", alpha_n.shape)    # torch.Size([128, 38, 38])
        # 对权重化的x求和
        # [batch, seq_len, seq_len]·[batch,seq_len, hidden_dim*2] = [batch,seq_len,hidden_dim*2] -> [batch, hidden_dim*2]
        context = torch.matmul(alpha_n, x).sum(1)

        return context

    def forward(self,data):
        
        input_ids = data["input_ids"].to(self.device)
        emb = self.embedding(input_ids)
        output,(h,c) = self.lstm(emb)
        output = self.ln(output)
        query = self.dropout(output)
        #output = output.permute([1,0,2])
        
        attention = self.attention_net(output,query)
        fc = self.fc(attention)
        return F.log_softmax(fc)
