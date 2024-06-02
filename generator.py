import sys
from language_model import N_gram_model,Interpolation
from tokenizer import Tokeniser

lm_type=sys.argv[1]
corp_path=sys.argv[2]

if lm_type=='i':
    sent=input("input sentence: ")
    print("output:")
    k=int(sys.argv[3])
    model=Interpolation(corp_path)
    tok=Tokeniser(sent)
    sentences=tok.convert()
    sent=['<start>']*2+sentences[0]
    context_tup=tuple(sent[-2:])
    print(model.lam1,model.lam2,model.lam3)
    x=tuple()
    probs_per_word=[]
    for word in model.params1[x]:
        a=model.lam1*((model.params1[x][word])/sum(model.params1[x].values()))
        if word in model.params2:
            a+=model.lam2*(model.params2[tuple([context_tup[1]])][word]/sum((model.params2[tuple([context_tup[1]])]).values()))
        else:
            a+=model.lam2*(1e-5)
        if word in model.params3:
            a+=model.lam3*(model.params3[context_tup][word]/sum((model.params3[context_tup]).values()))
        else:
            a+=model.lam3*(1e-5)
        probs_per_word.append(tuple([word,a]))
    sorted_items = sorted(probs_per_word, key=lambda x: x[1], reverse=True)
    i=0
    while k and i<len(sorted_items):
        print(f'{sorted_items[i][0]} {sorted_items[i][1]}')
        i+=1
        k-=1
elif lm_type=='r':
    n=int(sys.argv[3])
    sen_len=int(sys.argv[4])
    # random sentence generation
    model=N_gram_model(corp_path,n,1000)
    model.read_file()
    model.setup()
    model.train()
    keys_to_remove=['"',',','-','/']
    for key in keys_to_remove:
        (model.params[tuple(['<start>']*(n-1))]).pop(key, None)
    max_key =max(model.params[tuple(['<start>']*(n-1))], key=model.params[tuple(['<start>']*(n-1))].get)
    sent=['<start>']*(n-2)
    sent.append(max_key)
    i=0
    for i in range(sen_len):
        context_tup=tuple(sent[i:i+n-1])
        max_key =max(model.params[context_tup], key=model.params[context_tup].get)
        sent.append(max_key)
    print("Sentence formed : ", end='')
    for i in range(n-2,len(sent)):
        print(sent[i]+" ", end='')
elif lm_type=='n':
    sent=input("input sentence: ")
    print("output:")
    k=int(sys.argv[3])
    n=3
    if len(sys.argv)>4:
        n=int(sys.argv[4])
    model=N_gram_model(corp_path,n,1000)
    model.read_file()
    model.setup()
    model.train()
    tok=Tokeniser(sent)
    sentences=tok.convert()
    sent=['<start>']*(n-1)+sentences[0]
    context_tup=tuple(sent[-(n-1):])
    params=model.params
    if context_tup in params:
        probs_per_word={x:(params[context_tup][x]/sum((params[context_tup]).values())) for x in params[context_tup]}
        sorted_items = sorted(probs_per_word.items(), key=lambda x: x[1], reverse=True)
        i=0
        while k and i<len(sorted_items):
            print(f'{sorted_items[i][0]} {sorted_items[i][1]}')
            i+=1
            k-=1
    else:
        model=N_gram_model(corp_path,1,1000)
        model.read_file()
        model.setup()
        model.train()
        tup=tuple()
        lst={x:(model.params[tup][x]/sum((model.params[tup]).values())) for x in model.params[tup]}
        sorted_items = sorted(lst.items(), key=lambda x: x[1], reverse=True)
        for i in range(min(k, len(sorted_items))):
            key, value = sorted_items[i]
            print(f"{key} {value}")