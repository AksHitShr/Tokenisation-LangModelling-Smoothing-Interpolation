import sys
from tokenizer import Tokeniser
import random
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

class N_gram_model:
    def __init__(self,corp,n,test_size):
        self.N=n
        self.corp_path=corp
        self.test_size=test_size
    def read_file(self):
        with open(self.corp_path,"r") as file:
            x=file.read()
        self.corpus=x.lower()
    def setup(self):
        tok=Tokeniser(self.corpus)
        sentences=tok.convert()
        self.vocab=len(set([item for sublist in sentences for item in sublist]))
        self.corp_size=len([item for sublist in sentences for item in sublist])
        random.seed(12)
        random.shuffle(sentences)
        split_index = self.test_size
        self.test_data = sentences[:split_index]
        self.train_data = sentences[split_index:]
    def train(self):
        params={}
        for sen in self.train_data:
            mod_sen=["<start>"]*(self.N-1)+sen+["<end>"]
            for i in range(len(mod_sen) - self.N + 1):
                context_tup = tuple(mod_sen[i:i+self.N-1])
                last_word=mod_sen[i+self.N-1]
                if context_tup in params:
                    if last_word in params[context_tup]:
                        params[context_tup][last_word]+=1
                    else:
                        params[context_tup][last_word]=1
                else:
                    params[context_tup]={last_word:1}
        self.params=params

class good_turing:
    def perplexity(self,model,typ):
        perp=[]
        type_dt=model.train_data if typ==1 else model.test_data
        for sen in type_dt:
            sent=["<start>"]*(model.N-1)+sen+['<end>']
            probs=[]
            for i in range(len(sent) - model.N + 1):
                context_tup = tuple(sent[i:i+model.N-1])
                last_word=sent[i+model.N-1]
                if context_tup in model.params:
                    a=0
                    b=model.vocab
                    for inner_key in model.params[context_tup]:
                        a+=self.freq_of_freq[model.params[context_tup][inner_key]]
                        b-=1
                    c=0
                    if last_word in model.params[context_tup]:
                        c=self.freq_of_freq[model.params[context_tup][last_word]]
                    else:
                        c=self.freq_of_freq[0]
                    c/=(a+(b*self.freq_of_freq[0]))
                    probs.append(np.log(c))
                else:
                   probs.append(-np.log(model.vocab))
            y=-np.sum(probs)/len(sent)
            perp.append(np.exp(y))
        print(f'Perplexity Score: {np.average(perp)}')
    def __init__(self,cor,n):
        self.corp_path=cor
        model=N_gram_model(self.corp_path,n,1000)
        model.read_file()
        model.setup()
        model.train()
        freq_of_freq={}
        for outer_key, inner_dict in model.params.items():
            for inner_key, inner_value in inner_dict.items():
                if inner_value in freq_of_freq:
                    freq_of_freq[inner_value]+=1
                else:
                    freq_of_freq[inner_value]=1
        sorted_keys = sorted(freq_of_freq.keys())
        z_r=[]
        r=[]
        for i in range(len(sorted_keys)):
            q=sorted_keys[i-1] if i>0 else 0
            z=0
            if i==len(sorted_keys)-1:
                z=freq_of_freq[sorted_keys[i]]/(sorted_keys[i]-q)
            else:
                t=sorted_keys[i+1]
                z=freq_of_freq[sorted_keys[i]]/(0.5*(t-q))
            z_r.append(z)
            r.append(sorted_keys[i])
        logr=np.log(r).tolist()
        logzr=np.log(z_r).tolist()
        lin_reg=LinearRegression()
        r_nparr=np.array(logr).reshape(-1,1)
        lin_reg.fit(r_nparr,logzr)
        # plt.scatter(logr,logzr,label='original data')
        # plt.plot(logr,lin_reg.predict(np.array(logr).reshape(-1,1)),label='linear regression line')
        # plt.xlabel('log(r)')
        # plt.ylabel('log(Z_r)')
        # plt.legend()
        # plt.show()
        new_keys=range(sorted_keys[0],sorted_keys[-1]+2)
        to_pred=np.log(new_keys)
        temp=list(lin_reg.predict(to_pred.reshape(-1,1)))
        temp=np.exp(temp).tolist()
        temp_dict=dict(zip(new_keys,temp))
        new_freq_of_freq={0:temp_dict[1]}
        for i in range(len(sorted_keys)):
            new_freq_of_freq[sorted_keys[i]]=((sorted_keys[i]+1)*temp_dict[sorted_keys[i]+1])/temp_dict[sorted_keys[i]]
        self.freq_of_freq=new_freq_of_freq
        if len(sys.argv)>3 and sys.argv[3]=='p_test':
            self.perplexity(model,0)
        elif len(sys.argv)>3 and sys.argv[3]=='p_train':
            self.perplexity(model,1)
        elif len(sys.argv)>3 and sys.argv[3]=='file_test':
            perp=[]
            sen_perps=[]
            probs=[]
            for sen in model.test_data:
                to_add=[sen]
                sent=["<start>"]*(model.N-1)+sen+['<end>']
                probs=[]
                for i in range(len(sent) - model.N + 1):
                    context_tup = tuple(sent[i:i+model.N-1])
                    last_word=sent[i+model.N-1]
                    if context_tup in model.params:
                        a=0
                        b=model.vocab
                        for inner_key in model.params[context_tup]:
                            a+=self.freq_of_freq[model.params[context_tup][inner_key]]
                            b-=1
                        c=0
                        if last_word in model.params[context_tup]:
                            c=self.freq_of_freq[model.params[context_tup][last_word]]
                        else:
                            c=self.freq_of_freq[0]
                        c/=(a+(b*self.freq_of_freq[0]))
                        probs.append(np.log(c))
                    else:
                        probs.append(-np.log(model.vocab))
                y=-np.sum(probs)/len(sent)
                to_add.append(np.exp(y))
                sen_perps.append(to_add)
                perp.append(np.exp(y))
            with open('2021101029_LM3_test-perplexity.txt', 'w') as file:
                file.write(f'Average Perplexity: {np.average(perp)}\n')
                for t in sen_perps:
                    file.write(f'{t[0]}\t{t[1]}\n')
        elif len(sys.argv)>3 and sys.argv[3]=='file_train':
            perp=[]
            sen_perps=[]
            probs=[]
            for sen in model.train_data:
                to_add=[sen]
                sent=["<start>"]*(model.N-1)+sen+['<end>']
                probs=[]
                for i in range(len(sent) - model.N + 1):
                    context_tup = tuple(sent[i:i+model.N-1])
                    last_word=sent[i+model.N-1]
                    if context_tup in model.params:
                        a=0
                        b=model.vocab
                        for inner_key in model.params[context_tup]:
                            a+=self.freq_of_freq[model.params[context_tup][inner_key]]
                            b-=1
                        c=0
                        if last_word in model.params[context_tup]:
                            c=self.freq_of_freq[model.params[context_tup][last_word]]
                        else:
                            c=self.freq_of_freq[0]
                        c/=(a+(b*self.freq_of_freq[0]))
                        probs.append(np.log(c))
                    else:
                        probs.append(-np.log(model.vocab))
                y=-np.sum(probs)/len(sent)
                to_add.append(np.exp(y))
                sen_perps.append(to_add)
                perp.append(np.exp(y))
            with open('2021101029_LM3_train-perplexity.txt', 'w') as file:
                file.write(f'Average Perplexity: {np.average(perp)}\n')
                for t in sen_perps:
                    file.write(f'{t[0]}\t{t[1]}\n')
        elif len(sys.argv)<=3:
            freq_of_freq=new_freq_of_freq
            sent=input("input sentence: ")
            sent=sent.lower()
            tok=Tokeniser(sent)
            sentences=tok.convert()
            sent=sentences[0]
            sent=["<start>"]*2+sent+['<end>']
            probs=[]
            for i in range(len(sent) - model.N+1):
                context_tup = tuple(sent[i:i+model.N-1])
                last_word=sent[i+model.N-1]
                if context_tup in model.params:
                    a=0
                    b=model.vocab
                    for inner_key in model.params[context_tup]:
                        a+=freq_of_freq[model.params[context_tup][inner_key]]
                        b-=1
                    c=0
                    if last_word in model.params[context_tup]:
                        c=freq_of_freq[model.params[context_tup][last_word]]
                    else:
                        c=freq_of_freq[0]
                    c/=(a+(b*freq_of_freq[0]))
                    probs.append(np.log(c))
                else:
                    probs.append(-np.log(model.vocab))
            y=np.exp(np.sum(probs))
            print(f'score: {y}')

class Interpolation:
    def __init__(self,corp):
        corp_path=corp
        model3=N_gram_model(corp_path,3,1000)
        model3.read_file()
        model3.setup()
        model3.train()
        model2=N_gram_model(corp_path,2,1000)
        model2.read_file()
        model2.setup()
        model2.train()
        model1=N_gram_model(corp_path,1,1000)
        model1.read_file()
        model1.setup()
        model1.train()
        params3=model3.params
        params2=model2.params
        params1=model1.params
        lam1=0
        lam2=0
        lam3=0
        for outer_key in params3:
            sumall_outer=sum((params3[outer_key]).values())
            for inner_key in params3[outer_key]:
                if sumall_outer==1:
                    case1=0
                else:
                    case1=(params3[outer_key][inner_key]-1)/(sumall_outer-1)
                if sum((params2[tuple([outer_key[1]])]).values())==1:
                    case2=0
                else:
                    case2=(params2[tuple([outer_key[1]])][inner_key]-1)/(sum((params2[tuple([outer_key[1]])]).values())-1)
                x=tuple()
                case3=(params1[x][inner_key]-1)/(model2.corp_size-1)
                y=max(case1,case2,case3)
                if case1==y:
                    lam3+=params3[outer_key][inner_key]
                elif case2==y:
                    lam2+=params3[outer_key][inner_key]
                else:
                    lam1+=params3[outer_key][inner_key]
                q=lam1+lam2+lam3
                lam1=lam1/q
                lam2=lam2/q
                lam3=lam3/q
        self.lam1=lam1
        self.lam2=lam2
        self.lam3=lam3
        self.params1=params1
        self.params2=params2
        self.params3=params3
        if len(sys.argv)>3 and sys.argv[3]=='p_test':
            perp=[]
            for sen in model3.test_data:
                sent=["<start>"]*2+sen+['<end>']
                probs=[]
                for i in range(len(sent) - 2):
                    context_tup = tuple(sent[i:i+2])
                    last_word=sent[i+2]
                    x=tuple()
                    a=lam1*((params1[x][last_word])/sum(params1[x].values())) if last_word in params1[x] else 0
                    if tuple([context_tup[1]]) in params2:
                        if last_word in params2[tuple([context_tup[1]])]:
                            a+=lam2*(params2[tuple([context_tup[1]])][last_word]/sum((params2[tuple([context_tup[1]])]).values()))
                    if context_tup in params3:
                        if last_word in params3[context_tup]:
                            a+=lam3*(params3[context_tup][last_word]/sum((params3[context_tup]).values()))
                    if a==0:
                        a=1e-15
                    probs.append(np.log(a))
                y=-np.sum(probs)/len(sent)
                perp.append(np.exp(y))
            print(f'Perplexity Score: {np.average(perp)}')
        elif len(sys.argv)>3 and sys.argv[3]=='p_train':
            perp=[]
            for sen in model3.train_data:
                sent=["<start>"]*(2)+sen+['<end>']
                probs=[]
                for i in range(len(sent) - 2):
                    context_tup = tuple(sent[i:i+2])
                    last_word=sent[i+2]
                    x=tuple()
                    a=lam1*((params1[x][last_word])/sum(params1[x].values())) if last_word in params1[x] else lam1*1e-15
                    if tuple([context_tup[1]]) in params2:
                        if last_word in params2[tuple([context_tup[1]])]:
                            a+=lam2*(params2[tuple([context_tup[1]])][last_word]/sum((params2[tuple([context_tup[1]])]).values()))
                        else:
                            a+=lam2*1e-15
                    else:
                        a+=lam2*1e-15
                    if context_tup in params3:
                        if last_word in params3[context_tup]:
                            a+=lam3*(params3[context_tup][last_word]/sum((params3[context_tup]).values()))
                        else:
                            a+=lam2*1e-15
                    else:
                        a+=lam2*1e-15
                    probs.append(np.log(a))
                y=-np.sum(probs)/len(sent)
                perp.append(np.exp(y))
            print(f'Perplexity Score: {sum(perp)/(len(perp))}')
        elif len(sys.argv)>3 and sys.argv[3]=='file_train':
            perp=[]
            sen_perps=[]
            for sen in model3.train_data:
                to_add_later=[sen]
                sent=["<start>"]*(2)+sen+['<end>']
                probs=[]
                for i in range(len(sent) - 2):
                    context_tup = tuple(sent[i:i+2])
                    last_word=sent[i+2]
                    x=tuple()
                    a=lam1*((params1[x][last_word])/sum(params1[x].values())) if last_word in params1[x] else lam1*1e-15
                    if tuple([context_tup[1]]) in params2:
                        if last_word in params2[tuple([context_tup[1]])]:
                            a+=lam2*(params2[tuple([context_tup[1]])][last_word]/sum((params2[tuple([context_tup[1]])]).values()))
                        else:
                            a+=lam2*1e-15
                    else:
                        a+=lam2*1e-15
                    if context_tup in params3:
                        if last_word in params3[context_tup]:
                            a+=lam3*(params3[context_tup][last_word]/sum((params3[context_tup]).values()))
                        else:
                            a+=lam2*1e-15
                    else:
                        a+=lam2*1e-15
                    probs.append(np.log(a))
                y=-np.sum(probs)/len(sent)
                perp.append(np.exp(y))
                to_add_later.append(np.exp(y))
                sen_perps.append(to_add_later)
            # print(f'Perplexity Score: {sum(perp)/(len(sent)-2)}')
            with open('2021101029_LM2_train-perplexity.txt', 'w') as file:
                file.write(f'Average Perplexity: {np.average(perp)}\n')
                for t in sen_perps:
                    file.write(f'{t[0]}\t{t[1]}\n')
        elif len(sys.argv)>3 and sys.argv[3]=='file_test':
            perp=[]
            sen_perps=[]
            for sen in model3.test_data:
                to_add_later=[sen]
                sent=["<start>"]*(2)+sen+['<end>']
                probs=[]
                for i in range(len(sent) - 2):
                    context_tup = tuple(sent[i:i+2])
                    last_word=sent[i+2]
                    x=tuple()
                    a=lam1*((params1[x][last_word])/sum(params1[x].values())) if last_word in params1[x] else lam1*1e-15
                    if tuple([context_tup[1]]) in params2:
                        if last_word in params2[tuple([context_tup[1]])]:
                            a+=lam2*(params2[tuple([context_tup[1]])][last_word]/sum((params2[tuple([context_tup[1]])]).values()))
                        else:
                            a+=lam2*1e-15
                    else:
                        a+=lam2*1e-15
                    if context_tup in params3:
                        if last_word in params3[context_tup]:
                            a+=lam3*(params3[context_tup][last_word]/sum((params3[context_tup]).values()))
                        else:
                            a+=lam2*1e-15
                    else:
                        a+=lam2*1e-15
                    probs.append(np.log(a))
                y=-np.sum(probs)/len(sent)
                perp.append(np.exp(y))
                to_add_later.append(np.exp(y))
                sen_perps.append(to_add_later)
            # print(f'Perplexity Score: {sum(perp)/(len(sent)-2)}')
            with open('2021101029_LM2_test-perplexity.txt', 'w') as file:
                file.write(f'Average Perplexity: {np.average(perp)}\n')
                for t in sen_perps:
                    file.write(f'{t[0]}\t{t[1]}\n')
        elif len(sys.argv)<=3:
            sent=input("input sentence: ")
            tok=Tokeniser(sent)
            sentences=tok.convert()
            sent=sentences[0]
            sent=["<start>"]*2+sent+['<end>']
            prob=1
            for i in range(len(sent)-2):
                context_tup = tuple(sent[i:i+2])
                last_word=sent[i+2]
                x=tuple()
                a=lam1*((params1[x][last_word])/sum(params1[x].values())) if last_word in params1[x] else 1e-5
                if tuple([context_tup[1]]) in params2:
                    if last_word in params2[tuple([context_tup[1]])]:
                        a+=lam2*(params2[tuple([context_tup[1]])][last_word]/sum((params2[tuple([context_tup[1]])]).values()))
                if context_tup in params3:
                    if last_word in params3[context_tup]:
                        a+=lam3*(params3[context_tup][last_word]/sum((params3[context_tup]).values()))
                prob*=a
            print(f'score: {prob}')

if __name__=='__main__':
    corp_path=sys.argv[2]
    if sys.argv[1]=='g':
    # good turing
        curr_model=good_turing(corp_path,3)
    elif sys.argv[1]=='i':
        # interpolation
        curr_model=Interpolation(corp_path)
    elif sys.argv[1]=='n':
        # normal n-gram
        sent=input("input sentence: ")
        tok=Tokeniser(sent)
        sentences=tok.convert()
        sent=sentences[0]
        sent=["<start>"]*2+sent+['<end>']
        model=N_gram_model(corp_path,3,1000)
        model.read_file()
        model.setup()
        model.train()
        params=model.params
        prob=1
        for i in range(len(sent)-2):
            context_tup = tuple(sent[i:i+2])
            last_word=sent[i+2]
            if context_tup in params:
                if last_word in params[context_tup]:
                    prob*=params[context_tup][last_word]/sum((params[context_tup]).values())
                else:
                    prob*=1e-5
            else:
                prob*=1e-5
        print(f'score: {prob}')