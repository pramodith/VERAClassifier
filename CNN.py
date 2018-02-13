import torch
from torch import backends
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import dataloader
import numpy as np
from torch.autograd import Variable
from collections import defaultdict
import json
import time

class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()
        self.voacb_size=5304
        self.batch_size=1
        self.num_filters=3
        self.word_embed=torch.load("word2vec.pkl")
        self.in_channel=1
        self.filter_sizes=[1,2,3]
        self.embedding_length=200
        self.number_classes=4
        self.embed = nn.Embedding(self.voacb_size + 1, self.embedding_length)
        #self.embed.weight.data.copy_(torch.from_numpy(self.word_embed.weight.data))
        #self.embed.weight.requires_grad=False
        self.conv1=nn.ModuleList([nn.Conv2d(in_channels=self.in_channel,out_channels=self.num_filters,kernel_size=(x,self.embedding_length)) for x in self.filter_sizes])
        self.dropout=nn.Dropout(p=0.5)
        self.fc=nn.Linear(self.num_filters*len(self.filter_sizes),self.number_classes)


    def forward(self,input_x):
        x = self.embed(input_x)
        x=x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (batch_size,len(kernel_sizes)*num_kernels)
        logit = F.log_softmax(self.fc(x))  # (batch_size, num_aspects)
        return logit


#model=CNN()

def build_word_idx(filenames):
    word_to_index={}
    for file in filenames:
        with open(file,'r') as f:
            lines=f.read()
            lines=list(set(lines.split(" ")))
        for x in lines:
            word_to_index[x]=len(word_to_index)+1
    word_to_index=defaultdict(int,word_to_index)
    return word_to_index
word_to_index=build_word_idx(["Wikipedia/sharks_wikipedia.txt","Wikipedia/cheetahs.txt"])
print(len(word_to_index))
all_classes=["other","eats","habitat","lifespan"]

def create_dataset(filenames):
    X=[]
    y=[]
    ind=-1
    for file in filenames:
        with open(file,'r') as f:
            doc_dict=json.load(f)
        for cnt,sent in enumerate(doc_dict):
            X.append([word_to_index[ind] for ind in sent['x'].split(" ")])
            ind+=1
            while len(X[ind])<50:
                X[ind].append(5305)
            if len(X)>50:
                X[ind]=X[ind][:50]
            y.append(all_classes.index(sent['y']))
    return X,y

X,y=create_dataset(["datasets/sharks.json","datasets/cheetahs.json"])

def train_model(X,y,epochs=100):
    cnt=0
    model = CNN()
    loss_fn = nn.NLLLoss()
    loss_accumulated = 0
    model=model.cuda()
    #trainloader = torch.utils.data.DataLoader(X, batch_size=32, shuffle=True, num_workers=8)
    #trainloader_label=torch.utils.data.DataLoader(y, batch_size=32, shuffle=True, num_workers=8)
    optimizer = torch.optim.SGD(model.parameters(),lr=0.05,momentum=0.5)
    for i in range(epochs):
        for data,label in zip(X,y):
            data, target = Variable(torch.LongTensor([data])).cuda(), Variable(torch.LongTensor([label])).cuda()
            model.zero_grad()
            logit=model(data)
            cnt+=1
            loss=loss_fn(logit,target)
            loss.backward()
            loss_accumulated+=loss.data
            print(loss.data)
            optimizer.step()
        if i%10==0:
            print("Sleeping ...")
            time.sleep(5)
        print("Epoch is "+ str(i))
        print(loss_accumulated)
    torch.save(model.state_dict(), "CNN_params.pkl")

backends.cudnn.benchmark=True
train_model(X,y,100)
