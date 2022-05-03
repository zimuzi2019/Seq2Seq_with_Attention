#利用训练好的模型进行推理计算

import random
import torch
import torch.nn as nn
from torch import optim
from datasets import readLangs,SOS_token,EOS_token,MAX_LENGTH
from models import EncoderRNN,AttenDecoderRNN
from utils import timeSince,normalizeString,cht_to_chs
import time

import jieba

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH = MAX_LENGTH+1

lang1="en"
lang2="cn"

hidden_size=256

path="/home/cslee/MyPythonCode/12_demo0/data/train_en-cn.txt"
input_lang,output_lang,pairs=readLangs(lang1,lang2,path)

#print(len(pairs))
#print(input_lang.n_words)
#print(input_lang.index2word)

#print(output_lang.n_words)
#print(output_lang.index2word)


def listTotensor(input_lang,data):
    indexes_in=[input_lang.word2index[word] if word in input_lang.word2index else input_lang.word2index["UNK"] 
                for word in data.split(" ")] 

    indexes_in.append(EOS_token)     
    input_tensor=torch.tensor(indexes_in,
                              dtype=torch.long,
                              device=device).view(-1,1)
    return input_tensor

def tensorFromPair(pair):
    input_tensor=listTotensor(input_lang,pair[0])
    output_tensor=listTotensor(output_lang,pair[1])
    return (input_tensor,output_tensor)



#定义encoder和decoder网络
encoder=EncoderRNN(input_lang.n_words,hidden_size).to(device)
decoder=AttenDecoderRNN(hidden_size,
                        output_lang.n_words,
                        max_len=MAX_LENGTH,
                        dropout_p=0.1).to(device)

#加载参数
encoder.load_state_dict(torch.load("/home/cslee/MyPythonCode/12_demo0/models/encoder_700000.pth"))
decoder.load_state_dict(torch.load("/home/cslee/MyPythonCode/12_demo0/models/decoder_700000.pth"))





path="/home/cslee/MyPythonCode/12_demo0/data/test_en-cn.txt"
lines=open(path,encoding="utf-8").readlines()

import random
random.shuffle(lines)




test_batch,cnt,i,score_sum=30,0,0,0


from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
#from nltk.translate.bleu_score import SmoothingFunction


while True:
    l=lines[i].split("\t")

    sentence1=normalizeString(l[0])

    sentence2=cht_to_chs(l[1].strip())
    seg_list=jieba.cut(sentence2,cut_all=False)
    sentence2=" ".join(seg_list)

    #定义pair对
    train_sen_pair=[sentence1,sentence2]
    training_pair=tensorFromPair(train_sen_pair)

    input_tensor,output_tensor=training_pair

    if(len(output_tensor) > MAX_LENGTH or len(input_tensor) > MAX_LENGTH):
        i += 1
        continue
    
    #encoder inference
    encoder_hidden=encoder.initHidden()
    input_len=input_tensor.size(0)
    encoder_outputs=torch.zeros(MAX_LENGTH,encoder.hidden_size,device=device)

    for ei in range(input_len):
        encoder_output,encoder_hidden=encoder(input_tensor[ei],encoder_hidden)  
        encoder_outputs[ei]=encoder_output[0,0]
    
    #decoder
    decoder_hidden=encoder_hidden                         
    decoder_input=torch.tensor([[SOS_token]],device=device)
    use_teacher_forcing=True if random.random() < 0.5 else False

    decoder_words=[]

    ##由于做inference不知道输出的长度，这里将它设置为最大的长度
    for di in range(MAX_LENGTH):
        decoder_output,decoder_hidden,decoder_attention=decoder(
            decoder_input,decoder_hidden,encoder_outputs
        )
        topV,topi=decoder_output.topk(1)      
        decoder_input=topi.squeeze().detach()

        if topi.item() == EOS_token:
            decoder_words.append("<EOS>")
            break
        else:
            decoder_words.append(output_lang.index2word[topi.item()])



    print(train_sen_pair[0]," ---------- ",train_sen_pair[1])   #input
    print(train_sen_pair[1].split(" "))                         #output
    print(decoder_words[:-1])
  

    # #smooth = SmoothingFunction()  # 定义平滑函数对象
    # labels = ['我','是','谁']
    # predicts = ['我','是','猫']
    # corpus_score_2 = corpus_bleu(labels, predicts, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth.method1)
    # corpus_score_4 = corpus_bleu(labels, predicts, smoothing_function=smooth.method1)

    score_1 = sentence_bleu([train_sen_pair[1].split(" ")],decoder_words[:-1],weights=(1,0,0,0))
    score_2 = sentence_bleu([train_sen_pair[1].split(" ")],decoder_words[:-1],weights=(0,1,0,0))
    score_3 = sentence_bleu([train_sen_pair[1].split(" ")],decoder_words[:-1],weights=(0,0,1,0))
    score_4 = sentence_bleu([train_sen_pair[1].split(" ")],decoder_words[:-1],weights=(0,0,0,1))
    print(score_1,score_2,score_3,score_4)

    score_sum += sentence_bleu([train_sen_pair[1].split(" ")],decoder_words[:-1],weights=(0.5,0.5,0,0))
    print("\n")

    i += 1
    cnt += 1
    if cnt == test_batch:
        break

   
print("The avrage BLEU score of test batch is ",score_sum*1.0/test_batch)