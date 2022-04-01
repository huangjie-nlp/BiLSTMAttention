
class Config():
    def __init__(self):
        self.label_num = 15
        self.cuda = 0
        self.dropout = 0.5
        self.epoch = 20
        self.batch_size = 32
        self.learning_rate = 1e-3
        self.step = 1000
        self.unit = 128
        self.char_num = 6679
        self.word_num = 205513
        self.embed_dim = 128
        self.label_flag = "stock"
        self.bert_dim = 768
        self.train_fn = "./dataset/train_data.csv"
        self.dev_fn = "./dataset/dev_data.csv"
        self.test_fn = "./dataset/test_data.csv"
        self.schema_fn = "./dataset/schema.json"
        self.log = "./log/{}_log.log"
        self.flag = "word"
        self.save_model = "./checkpoint/BiLSTM_Attention_{}.pt".format(self.flag)
        self.dev_result = "./dev_result/dev.csv"
        self.test_result = "./test_result/test.csv"
        self.char_vocab = "./dataset/char_vocab.json"
        self.word_vocab = "./dataset/word_vocab.json"
        
        
