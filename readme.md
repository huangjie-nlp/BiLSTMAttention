# BiLSTMAttention
使用BiLSTM+Attention的方法 进行文本分类

## 说明
1、config/config.py是模型的配置文件（配置文件中可以设置flag为char则以字来切割句子，flag设置为word则以词来切分句子，label_flag设置负类标签）  
2、模型训练好后保存在checkpoint文件夹下  
3、Python main.py直接训练模型
4、python test.py单句预测

## 实验环境
torch == 1.7.1  
jieba == 0.42.1  
pandas == 1.1.5  
tqdm == 4.59.0  
