#定义主干的模型结构。主要包含两个模块，一个是编码模块，一个是解码模块

#在编码模块，通过RNN网络来进行编码
#在解码模块，引入Attention结构

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import MAX_LENGTH

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#定义编码的RNN结构
class EncoderRNN(nn.Module):
    def __init__(self,input_size,hidden_size): #分别定义输入的句子所对应的长度以及隐藏层结点的长度
        super(EncoderRNN,self).__init__()
        self.hidden_size=hidden_size

        #定义embedding层
        #输入句子的长度，输出对应编码成词向量所对应的长度，这里同样传入hidden_size
        self.embedding=nn.Embedding(input_size,hidden_size)

        #定义RNN网络，采用LSTM也可以
        #输入长度是embedding之后提取出的词向量，词向量的长度就是hidden_size对应的长度，隐藏层结点的size同样传入hidden_size
        self.gru=nn.GRU(hidden_size,hidden_size)


    def forward(self,input,hidden):  #传入对应的输入的数据和隐藏层的结点       
        #将编码之后的tensor进行维度的转换，将编码之后的这个维度传入到最后一个维度上去
        embedded=self.embedding(input).view(1,1,-1)
        output=embedded 
        output,hidden=self.gru(output,hidden)
        return output,hidden

    #定义函数对hidden进行初始化，因为在第一个结点上实际上需要定义一个默认的隐藏层结点
    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size,device=device)


#定义基于Attention的解码RNN结构
class AttenDecoderRNN(nn.Module):
    def __init__(self,hidden_size,output_size,dropout_p=0.1,max_len=MAX_LENGTH):
        super(AttenDecoderRNN,self).__init__()

        #对类内变量隐藏层结点数量、输出结点尺寸、dropout对应参数、序列最大长度max_len进行赋值
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.dropout_p=dropout_p
        self.max_len=max_len

        self.embedding=nn.Embedding(self.output_size,self.hidden_size)
        
        #定义Attention模块，实际上就是一个线性层
        #后面的结果要对embedding的结果和GRU网络的结果进行concat网络连接，所以传入hidden_size*2。第二个参数是序列的最大长度max_len
        #利用网络来拿到一个max_len长度的权重值，这个权重值会作用到序列的信息上去进行加权处理
        self.attn=nn.Linear(self.hidden_size*2,self.max_len)
        self.attn_combine=nn.Linear(self.hidden_size*2,self.hidden_size)
       
        #加入dropout，尽量减少过拟合情况的出现
        self.dropout=nn.Dropout(self.dropout_p)

        self.gru=nn.GRU(self.hidden_size,self.hidden_size)
        self.out=nn.Linear(self.hidden_size,self.output_size)


    #定义好基本的网络模块组件后，拼接组件拿到最终的网络
    def forward(self,input,hidden,encoder_outputs):
        #调用embedding，对输入序列进行特征抽取，将它转化成词向量
        embedded=self.embedding(input).view(1,1,-1)
        embedded=self.dropout(embedded)


        #计算attention的权重，通过softmax对权重进行归一化处理
        atten_weight=F.softmax(
            #调用attention，将embedding和hidden这两个tensor信息在第1维上拼接后作为attention的输入，以此来学对应的权重
            self.attn(torch.cat([embedded[0],hidden[0]],1)),
            dim=1
        )

        #将权重作用到feature上去。直接对权重和encoder之后的输出进行矩阵乘运算
        att_applied=torch.bmm(
            atten_weight.unsqueeze(0),
            encoder_outputs.unsqueeze(0)
        )

        #将经过加权之后的feature和embedding层进行concat，拿到下一层的输出
        output=torch.cat([embedded[0],att_applied[0]],dim=1)

        #对output利用attn_combine再进行一次计算，加入一个线性层
        output=self.attn_combine(output).unsqueeze(0)
        output=F.relu(output)
        output,hidden=self.gru(output,hidden)
        output=F.log_softmax(self.out(output[0]),dim=1)

        return output,hidden,atten_weight

    #同样对于第一个隐藏层需要进行初始化
    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size,device=device)



if __name__ == "__main__":
    encoder_net=EncoderRNN(5000,256)   #输入结点的size实际上就是字典的长度，隐藏层结点的size
    atten_decoder_net=AttenDecoderRNN(256,5000)

    #模拟输入
    tensor_in=torch.tensor([12,14,16,18],dtype=torch.long).view(-1,1)
    #定义隐藏层
    hidden_in=torch.zeros(6,1,256)

    #调用encoder_net进行编码。输出包括真正的输出和隐藏层编码信息
    #这里对tensor的第0个进行编码，实际训练时会通过一个for循环对tensor中的每一项分别进行编码。在编码时上一层的隐藏层会作为当前层的输入，而第一层会初始化一个隐藏层
    encoder_out,encoder_hidden=encoder_net(tensor_in[0],hidden_in)

    print(encoder_out)
    print(encoder_hidden)


    

    #构造解码网络输入和隐藏层
    tensor_in=torch.tensor([100]).view(-1,1)
    hidden_in = torch.zeros(1, 1, 256)

    #定义encoder输出
    encoder_out=torch.zeros(10, 256) #第二个参数256是确定的，和隐藏层的维度一样。第一个参数取决于在定义AttenDecoderRNN时定义的MAX_LEN，这里默认是15

    out1,out2,out3=atten_decoder_net(tensor_in,hidden_in,encoder_out)
    print(out1,out2,out3)