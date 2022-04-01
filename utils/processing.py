
import json
import jieba
import pandas as pd

def generate_schema(fn):
    df = pd.read_csv(fn)
    sentence = df.sentence.tolist()
    label = df.label.tolist()
    label2id,id2label = {},{}
    for l in label:
        if l not in label2id:
            label2id[l] = len(label2id)
            id2label[len(id2label)] = l
    return label2id,id2label

def generate_char_vocab(fn):
    df = pd.read_csv(fn)
    sentence = df.sentence.tolist()
    char2id = {"pad":0,"unk":1}
    for line in sentence:
        for char in line:
            if char not in char2id:
                char2id[char] = len(char2id)
    return char2id

def generate_word_vocab(fn):
    df = pd.read_csv(fn)
    sentence = df.sentence.tolist()
    word2id = {"pad":0,"unk":1}
    for line in sentence:
        seg = jieba.lcut(line)
        for char in seg:
            if char not in word2id:
                word2id[char] = len(word2id)
    return word2id

if __name__ == '__main__':
    file = "../dataset/train_data.csv"
    # label2id, id2label = generate_schema(file)
    # json.dump([label2id,id2label],open("../dataset/schema.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)
    char2id = generate_char_vocab(file)
    json.dump(char2id,open("../dataset/char_vocab.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)
    word2id = generate_word_vocab(file)
    json.dump(word2id, open("../dataset/word_vocab.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)
