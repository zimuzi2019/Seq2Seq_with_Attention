#在训练模块利用已经完成的模型和数据搭建整个训练任务
import random
import torch
import torch.nn as nn
from torch import optim
from datasets import readLangs,SOS_token,EOS_token,UNK_token,MAX_LENGTH
from models import EncoderRNN,AttenDecoderRNN
from utils import timeSince
import time

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_LENGTH=MAX_LENGTH+1  #最大长度定义为在dataset中MAX_LENGTH的长度加1，因为会插入一个结尾符

#通过调用readLangs函数 对数据进行读取
#完成从英文到中文的翻译
lang1="en"
lang2="cn"
path="/home/cslee/MyPythonCode/12_demo0/data/train_en-cn.txt"

#拿到输入语言类，输出语言类以及样本
input_lang,output_lang,pairs=readLangs(lang1,lang2,path)

# print(len(pairs))
# print(input_lang.n_words)
# print(input_lang.index2word)

# print(output_lang.n_words)
# print(output_lang.index2word)


#将pairs下的序列转化成输入的tensor，并且在tensor中插入终止符
def listTotensor(input_lang,data):
    #生成列表，列表中对应到输入句子所对应词的索引
    indexes_in=[input_lang.word2index[word] for word in data.split(" ")] 
    #在索引中加入终止符，这个是在训练attention模型时用到的终止符，终止符表示预测可以结束了，让Seq2Seq+Attention能够支撑变长序列的预测
    indexes_in.append(EOS_token)     
    input_tensor=torch.tensor(indexes_in,
                              dtype=torch.long,
                              device=device).view(-1,1)
    return input_tensor

def tensorFromPair(pair):
    input_tensor=listTotensor(input_lang,pair[0])
    output_tensor=listTotensor(output_lang,pair[1])
    return (input_tensor,output_tensor)



#搭建整个loss_function的网络，这里将encoder和decoder串起来做inference拿到最终的输出，然后和label计算loss
def loss_func(input_tensor,
              output_tensor,
              encoder,
              decoder,
              encoder_optimizer,
              decoder_optimizer,
              criterion):
    #首先定义一个隐藏层，这个是encoder的一个输入，调用encoder网络下的initHidden()初始化一个隐藏层
    encoder_hidden=encoder.initHidden()
    
    #将两个优化器梯度置0
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    #拿到输入和输出的长度
    input_len=input_tensor.size(0)
    output_len=output_tensor.size(0)

    #定义一个encoder的输出tensor
    encoder_outputs=torch.zeros(MAX_LENGTH,encoder.hidden_size,device=device)

    #定义编码过程
    for ei in range(input_len):
        encoder_output,encoder_hidden=encoder(input_tensor[ei],encoder_hidden)  
        encoder_outputs[ei]=encoder_output[0,0]

    #定义解码过程
    decoder_hidden=encoder_hidden                           #输入的隐藏层信息和encoder的定义方式相同
    decoder_input=torch.tensor([[SOS_token]],device=device) #将第一个解码的输入定义为起始符

    #加入teacher因子，通过对这个变量来进行随机来调整训练过程，为了加快网络收敛，加入teacher因子，
    #以0.5的概率进行随机随机地修改我们当前层预测的这个输入结果，将输入结果随机地修改为真实的label
    use_teacher_forcing=True if random.random() < 0.5 else False

    loss=0
    if use_teacher_forcing:
        for di in range(output_len):
            decoder_output,decoder_hidden,decoder_attention=decoder(
                decoder_input,decoder_hidden,encoder_outputs
            )

            loss += criterion(decoder_output,output_tensor[di])
            decoder_input=output_tensor[di]        #下一次循环的输入直接定义为output_tensor对应的内容，也就是label
    else:
        for di in range(output_len):
            decoder_output,decoder_hidden,decoder_attention=decoder(
                decoder_input,decoder_hidden,encoder_outputs
            )

            loss += criterion(decoder_output,output_tensor[di])
            topV,topi=decoder_output.topk(1)      #定义decoder下一次的输入为当前的预测结果
            decoder_input=topi.squeeze().detach()

            if decoder_input.item() == EOS_token: #如果decoder_input等于终止符，解码结束break
                break        
    
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / output_len




#定义训练部分代码
hidden_size=256 #定义隐藏层结点数量

#定义解码器和编码器

encoder=EncoderRNN(input_lang.n_words,hidden_size).to(device)
decoder=AttenDecoderRNN(hidden_size, 
                        output_lang.n_words,
                        max_len=MAX_LENGTH,
                        dropout_p=0.1).to(device)

#定义优化器
lr=0.01
encoder_optimizer=optim.SGD(encoder.parameters(),lr=lr)
decoder_optimizer=optim.SGD(decoder.parameters(),lr=lr)

#定义学习率调整的策略，每经过一次epoch之后会进行一次学习率的调整
scheduler_encoder=torch.optim.lr_scheduler.StepLR(encoder_optimizer,
                                                  step_size=1,
                                                  gamma=0.95)

scheduler_decoder=torch.optim.lr_scheduler.StepLR(decoder_optimizer,
                                                  step_size=1,
                                                  gamma=0.95)

#定义损失函数
criterion=nn.NLLLoss()

#这里直接根据最大的迭代次数生成对应样本对的数量，文本数据的整个数据量也不是很大

#定义最大的迭代次数
n_iters=700000 
#n_iters=100

training_pairs=[
    tensorFromPair(random.choice(pairs)) for i in range(n_iters)  #随机从样本对中选出样本
]   

print_every=100 
#print_every=10
save_every=100000

print_loss_total=0
start=time.time()

for iter in range(1,n_iters+1):
    training_pair=training_pairs[iter-1]
    input_tensor=training_pair[0]
    output_tensor=training_pair[1]

    loss=loss_func(input_tensor,
                   output_tensor,
                   encoder,
                   decoder,
                   encoder_optimizer,
                   decoder_optimizer,
                   criterion)

    print_loss_total += loss
    #print(iter,print_loss_total) #
    if iter % print_every == 0:
        print_loss_avg=print_loss_total/print_every
        print_loss_total=0
        print("{},{},{},{}".format(timeSince(start,iter/n_iters),
                                   iter,
                                   iter/n_iters*100,
                                   print_loss_avg))
    

    #定义保存模型和更新学习率的函数
    if iter % save_every == 0:
        torch.save(encoder.state_dict(),
                   "/home/cslee/MyPythonCode/12_demo0/models/encoder_{}.pth".format(iter))
        torch.save(decoder.state_dict(),
                   "/home/cslee/MyPythonCode/12_demo0/models/decoder_{}.pth".format(iter))
    
    #每训练10000次调整学习率
    if iter % 10000 == 0:
        scheduler_encoder.step()
        scheduler_decoder.step()