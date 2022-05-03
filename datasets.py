#数据处理模块 —— 完成对文本信息的读取和解析以及一些预处理
#在预处理过程中，会将中文进行分词，英文进行单词和标点的切分，并且将它们统计到一个字典中，基于字典对这些词进行编码

import jieba
from utils import normalizeString
from utils import cht_to_chs

#这里有两种语言，首先定义通用的一个方法，将这些方法定义在一个类里面，主要用来对不同的语言文本进行统计，完成从词转索引以及字典统计的功能

#定义两个token，分别对应到起始符和终止符
UNK_token=0
SOS_token=1
EOS_token=2

#定义字符串最大长度，太长的句子直接过滤掉
#因为在训练的时候需要将句子进行编码处理，如果句子太长，对整个训练任务会增加难度。而且如果句子中长句数量比较少，会存在分布不平衡的问题
MAX_LENGTH=10

class Lang:
    def __init__(self,name):
        self.name=name                #通过name来定义当前的语言是中文还是英文
        self.word2index={"UNK":0}     #对单词词语进行编码
        self.word2cont={"UNK":0}      #统计字典中每一个词出现的频率
        self.index2word={
            0:"UNK",1:"SOS",2:"EOS"   #首先定义未知字符，起始符和终止符
        }     #定义索引所对应的词，和word2index是对应关系
        self.n_words=3                   #统计当前语料库中有多少个单词。因为已经放了3个符号，所以初始值定为3
    
    def addWord(self,word):
        #对词进行统计，利用word来更新上面的几个字典值
        #给每个词一个索引值
        if word not in self.word2index:
            self.word2index[word]=self.n_words
            self.word2cont[word]=1
            self.index2word[self.n_words]=word
            self.n_words += 1
        else:
            self.word2cont[word] += 1

    def addSentence(self,sentence):  
        #对句子进行解析，将句子进行分词，分词之后将它放入到word中
        for word in sentence.split(" "):   #句子通过空格来拼接对应的词，这里通过空格进行切分
            self.addWord(word)

#定义文本解析的方法
def readLangs(lang1,lang2,path):      #传入两种文本，分别是中文和英文
    lines=open(path,encoding="utf-8").readlines()

    #定义两个类，分别对应到两种语言
    lang1_cls=Lang(lang1)
    lang2_cls=Lang(lang2)

    #对每一条数据进行拆分，拿到样本对
    pairs=[]
    for l in lines:
        l=l.split("\t")
        sentence1=normalizeString(l[0])  #英文
        sentence2=cht_to_chs(l[1].strip())
        seg_list=jieba.cut(sentence2,cut_all=False)
        sentence2=" ".join(seg_list)     #中文

        #对拿到的这两种语言对应的句子分别通过调用addSentence来生成字典结构以及词编码信息
        if len(sentence1.split(" "))>MAX_LENGTH:
            continue
        if len(sentence2.split(" "))>MAX_LENGTH:
            continue

        pairs.append([sentence1,sentence2])
        lang1_cls.addSentence(sentence1)
        lang2_cls.addSentence(sentence2)

    return lang1_cls,lang2_cls,pairs


if __name__ == "__main__":

    lang1="en"
    lang2="cn"
    path="/home/cslee/MyPythonCode/12_demo0/data/train_en-cn.txt"

    lang1_cls,lang2_cls,pairs=readLangs(lang1,lang2,path)


    print(lang1_cls.n_words)
    print(lang1_cls.word2index)

    print(lang2_cls.n_words)
    print(lang2_cls.index2word)

    print(len(pairs))
