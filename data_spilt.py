import random
import os 

path="/home/cslee/MyPythonCode/12_demo0/data/en-cn.txt"

lines=open(path,encoding='utf-8').readlines()
lines=[line.strip().split("\t") for line in lines]
random.shuffle(lines)


rate=0.998
tmp=int(len(lines)*rate)

train_lines=lines[0:tmp]
test_lines=lines[tmp:len(lines)]


if not os.path.exists("/home/cslee/MyPythonCode/12_demo0/data"):
    os.makedirs("/home/cslee/MyPythonCode/12_demo0/data")

save_train_path="/home/cslee/MyPythonCode/12_demo0/data/train_en-cn.txt"
save_test_path="/home/cslee/MyPythonCode/12_demo0/data/test_en-cn.txt"

f_train=open(save_train_path,'w')
for line in train_lines:
    f_train.write(line[0]+"\t"+line[1])
    f_train.write("\n")
f_train.close()

f_test=open(save_test_path,'w')
for line in test_lines:
    f_test.write(line[0]+"\t"+line[1])
    f_test.write("\n")
f_test.close()
