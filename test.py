import json
import re
import argparse
import torch
import model
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser(description = 'CNN1')
parser.add_argument('-embed-dim',type = int,default = 5,help = 'embedding_dim')#定义词嵌入向量的维度
parser.add_argument('-kernel-num',type = int,default = 100)#卷积核个数
parser.add_argument('-dropout',type = float,default = 0.5)
parser.add_argument('-batch-size',type = int, default = 1)#batch_size，因为我现在只用一条数据实验所以是1
parser.add_argument('-kernel-sizes',type = str,default = '3,4,5')#3个卷积核的形状
args = parser.parse_args()

#定义词典类
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.length = 0
    def add_word(self,word):
        if word not in self.idx2word:
            self.idx2word.append(word)
            self.word2idx[word] = self.length + 1
            self.length += 1
        return self.word2idx[word]
    def __len__(self):
        return len(self.idx2word)
    def onehot_encoded(self,word):
        vec = np.zeros(self.length)
        vec[self.word2idx[word]] = 1
        return vec

with open("./result/6.json",'r') as f:
    text = json.load(f)
    text1 = text['Major Surgical or Invasive Procedure']
    list1 = []
    print(type(text1))
    for i in text1:
        list1.extend(i['text'].split())
    print(list1)
#这是把6.json里面Major Surgical or Invasive Procedure的文本拿出来，组成了一个列表
#out:['Exploratory', 'right', '&', 'sigmoid', 'colectomy', 'small', 'bowel', 'resection', 'Abdominal', 'washout', 'Completion', 'colectomy']
    
    dic = Dictionary()
    for tok in list1:
        dic.add_word(tok)
    print(dic.word2idx)
#out:{'Exploratory': 1, 'right': 2, '&': 3, 'sigmoid': 4, 'colectomy': 5, 'small': 6, 'bowel': 7, 'resection': 8, 'Abdominal': 9, 'washout': 10, 'Completion': 11}

    list1_num = [0]*10 #限定长度为10，把句子里的单词转换成dic里的数字下标
    for i in range(10):
        list1_num[i] = dic.word2idx[list1[i]]
    print(list1_num)
#out:[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    feature = torch.LongTensor(list1_num)#建立LongTensor,准备输入模型
   # feature.data.t_()

    args.embed_num = dic.__len__()
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    cnn = model.textCNN(args)
    target = cnn(feature)
    print(target)
    
