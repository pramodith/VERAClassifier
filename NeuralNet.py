import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import download
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
import time
from collections import defaultdict

class CBOW(nn.Module):
    def __init__(self,context_size=2, embedding_size=50, vocab_size=None):
        super(CBOW,self).__init__()
        self.embeddings = nn.Embedding(vocab_size+1, embedding_size)
        self.linear1 = nn.Linear(embedding_size, vocab_size+1)

    def forward(self, inputs):
        lookup_embeds = self.embeddings(inputs)
        embeds = lookup_embeds.sum(dim=0)
        out = self.linear1(embeds)
        out = F.log_softmax(out)
        return out

    def make_context_vector(self,context,word_to_index):
        indexes=[word_to_index[x] for x in context]
        tensor = torch.LongTensor(indexes)
        if torch.cuda.is_available():
            tensor=tensor.cuda()
        return autograd.Variable(tensor)

class NeuralNet():

    Context_size=2
    EMBEDDING_SIZE=50

    def __init__(self):
        self.vocab=None
        self.word_set=set()
        self.words=[]
        self.X=[]
        self.word_to_index={}
        self.index_to_word={}
        self.embeddings=None
        self.linear1=None

    def preprocess(self,filename):
        with open(filename,'r') as f:
             wiki_text=f.read()

        stopWords = set(stopwords.words('english'))
        words = word_tokenize(wiki_text)
        wordsFiltered = []

        for w in words:
            if w not in stopWords:
                wordsFiltered.append(w)

    def file_to_set(self,filename):
        with open(filename,'r')as f:
            word_set=f.read().split(" ")
        return word_set

    def word_to_index_convert(self,files_list):
        for file in files_list:
            raw_words=(self.file_to_set(file))
            self.words.extend(raw_words)
            self.word_set=self.word_set.union(set(raw_words))
        self.vocab=len(self.word_set)
        word_to_index={word:cnt+1 for cnt,word in enumerate(list(self.word_set))}
        self.word_to_index=word_to_index
        return word_to_index

    def index_to_word(self,word_to_index):
        index_to_word={word_to_index[x]:x for x in word_to_index.keys()}
        index_to_word=defaultdict(int,index_to_word)
        self.index_to_word=index_to_word
        return index_to_word


    def create_dataset(self):
        global Context_size
        Context_size=2
        for i in range(Context_size,len(self.words)-Context_size):
            data=[self.words[i-2],self.words[i-1],self.words[i+1],self.words[i+2]]
            target=self.words[i]
            self.X.append((data,target))

n=NeuralNet()
n.word_to_index_convert(["Wikipedia/sharks_wikipedia.txt","Wikipedia/cheetahs.txt"])
n.create_dataset()

def train_model():
    Context_size=2
    EMBEDDING_SIZE=50
    loss_func = nn.CrossEntropyLoss()
    module=CBOW(Context_size, embedding_size=EMBEDDING_SIZE, vocab_size=n.vocab)
    module=    module.cuda()
    optimizer = optim.SGD(module.parameters(), lr=0.01)
    for epoch in range(100):
        total_loss = 0
        if epoch%10==0:
            time.sleep(5)
        for context, target in n.X:
            context_var = module.make_context_vector(context,n.word_to_index)
            module.zero_grad()
            log_probs=module(context_var)
            if not torch.cuda.is_available():
                loss = loss_func(log_probs.unsqueeze(0), autograd.Variable(
                torch.LongTensor([n.word_to_index[target]])))
            else:
                loss = loss_func(log_probs.unsqueeze(0), autograd.Variable(
                    torch.LongTensor([n.word_to_index[target]]).cuda()))
            loss.backward()
            optimizer.step()
            total_loss += loss.data
        print("Epoch is "+str(epoch))
        print(total_loss)
    torch.save(module.state_dict(),"word2vec_50.pkl")
train_model()

