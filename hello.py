import email
import re
import enchant
import codecs
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn import datasets
from sklearn.naive_bayes import *
from sklearn.metrics import accuracy_score
from nltk.stem import *

def create_clean(news):
    raw=open("data_files"+str(news),"r", encoding="ISO-8859-1")
    cleaned=open("data_filesv2"+str(news),"w")
    processed=raw.read()
    processed=processed.lower()
    processed=re.sub(r'\'s', r' ', processed)
    processed=re.sub(r'[0-9]+', r' ', processed)
    processed=re.sub(r'[\%\(\)\.\,\;\:\"\-]', r' ', processed)
    processed=re.sub(r'\s+', r' ', processed)
    cleaned.write(processed)
    #print("news:"+processed)

def preprocessing():
    for i in range(1,511):
        string="/business/"
        if(len(str(i))==1):
            string+="00"+str(i)
        elif(len(str(i))==2):
            string+="0"+str(i)
        else:
            string+=str(i)
        string+=".txt"
        create_clean(string)
    for i in range(1,387):
        string="/entertainment/"
        if(len(str(i))==1):
            string+="00"+str(i)
        elif(len(str(i))==2):
            string+="0"+str(i)
        else:
            string+=str(i)
        string+=".txt"
        create_clean(string)
    for i in range(1,418):
        string="/politics/"
        if(len(str(i))==1):
            string+="00"+str(i)
        elif(len(str(i))==2):
            string+="0"+str(i)
        else:
            string+=str(i)
        string+=".txt"
        create_clean(string)
    for i in range(1,512):
        string="/sport/"
        if(len(str(i))==1):
            string+="00"+str(i)
        elif(len(str(i))==2):
            string+="0"+str(i)
        else:
            string+=str(i)
        string+=".txt"
        create_clean(string)
    for i in range(1,401):
        string="/tech/"
        if(len(str(i))==1):
            string+="00"+str(i)
        elif(len(str(i))==2):
            string+="0"+str(i)
        else:
            string+=str(i)
        string+=".txt"
        create_clean(string)

def create_dictionary(d):
    dictionary=set()
    for i in range(1,511):
        string="data_filesv2/business/"
        if(len(str(i))==1):
            string+="00"+str(i)
        elif(len(str(i))==2):
            string+="0"+str(i)
        else:
            string+=str(i)
        string+=".txt"
        raw=open(string, "r")
        stremail=raw.read()
        words=stremail.split()
        for i in words:
            if(i[0]=='-'):
                continue
            if(i[len(i)-1]=='-'):
                continue
            if(i[len(i)-1]=='S'):
                if(i[len(i)-2]=='\''):
                    i=i[:-2]
            if d.check(i):
                dictionary.add(i)
    for i in range(1,387):
        string="data_filesv2/entertainment/"
        if(len(str(i))==1):
            string+="00"+str(i)
        elif(len(str(i))==2):
            string+="0"+str(i)
        else:
            string+=str(i)
        string+=".txt"
        raw=open(string, "r")
        stremail=raw.read()
        words=stremail.split()
        for i in words:
            if(i[0]=='-'):
                continue
            if(i[len(i)-1]=='-'):
                continue
            if(i[len(i)-1]=='S'):
                if(i[len(i)-2]=='\''):
                    i=i[:-2]
            if d.check(i):
                dictionary.add(i)
    for i in range(1,418):
        string="data_filesv2/politics/"
        if(len(str(i))==1):
            string+="00"+str(i)
        elif(len(str(i))==2):
            string+="0"+str(i)
        else:
            string+=str(i)
        string+=".txt"
        raw=open(string, "r")
        stremail=raw.read()
        words=stremail.split()
        for i in words:
            if(i[0]=='-'):
                continue
            if(i[len(i)-1]=='-'):
                continue
            if(i[len(i)-1]=='S'):
                if(i[len(i)-2]=='\''):
                    i=i[:-2]
            if d.check(i):
                dictionary.add(i)
    for i in range(1,512):
        string="data_filesv2/sport/"
        if(len(str(i))==1):
            string+="00"+str(i)
        elif(len(str(i))==2):
            string+="0"+str(i)
        else:
            string+=str(i)
        string+=".txt"
        raw=open(string, "r")
        stremail=raw.read()
        words=stremail.split()
        for i in words:
            if(i[0]=='-'):
                continue
            if(i[len(i)-1]=='-'):
                continue
            if(i[len(i)-1]=='S'):
                if(i[len(i)-2]=='\''):
                    i=i[:-2]
            if d.check(i):
                dictionary.add(i)
    for i in range(1,401):
        string="data_filesv2/tech/"
        if(len(str(i))==1):
            string+="00"+str(i)
        elif(len(str(i))==2):
            string+="0"+str(i)
        else:
            string+=str(i)
        string+=".txt"
        raw=open(string, "r")
        stremail=raw.read()
        words=stremail.split()
        for i in words:
            if(i[0]=='-'):
                continue
            if(i[len(i)-1]=='-'):
                continue
            if(i[len(i)-1]=='S'):
                if(i[len(i)-2]=='\''):
                    i=i[:-2]
            if d.check(i):
                dictionary.add(i)
    file=open("dictionary.txt", "w")
    wordlist=list(dictionary)
    wordlist.sort()
    for i in wordlist:
        file.write(i+"\n")

def check_contents(dictionary, current):
    file=open(current, "r")
    raw=file.read()
    words=raw.split()
    instance=[]
    string=""
    for i in dictionary:
        n=words.count(i)
        instance.append(n)
        string+=(str(n)+",")
    return (instance, string)

def create_feature_tr():
    dictionary=[]
    feature_vector=[]
    file=open("dictionary.txt", "r")
    feature_train=open("dataset-training.csv", "w")
    for i in file.readlines():
        i=i.strip()
        dictionary.append(i)

    for i in range(1,307):
        string="data_filesv2/business/"
        if(len(str(i))==1):
            string+="00"+str(i)
        elif(len(str(i))==2):
            string+="0"+str(i)
        else:
            string+=str(i)
        string+=".txt"
        current=string
        print("creating feature vector for"+current+"...")
        tuplee=check_contents(dictionary, current)
        feature_train.write(tuplee[1][:-1]+"\n")
    for i in range(1,233):
        string="data_filesv2/entertainment/"
        if(len(str(i))==1):
            string+="00"+str(i)
        elif(len(str(i))==2):
            string+="0"+str(i)
        else:
            string+=str(i)
        string+=".txt"
        current=string
        print("creating feature vector for"+current+"...")
        tuplee=check_contents(dictionary, current)
        feature_train.write(tuplee[1][:-1]+"\n")
    for i in range(1,251):
        string="data_filesv2/politics/"
        if(len(str(i))==1):
            string+="00"+str(i)
        elif(len(str(i))==2):
            string+="0"+str(i)
        else:
            string+=str(i)
        string+=".txt"
        current=string
        print("creating feature vector for"+current+"...")
        tuplee=check_contents(dictionary, current)
        feature_train.write(tuplee[1][:-1]+"\n")
    for i in range(1,308):
        string="data_filesv2/sport/"
        if(len(str(i))==1):
            string+="00"+str(i)
        elif(len(str(i))==2):
            string+="0"+str(i)
        else:
            string+=str(i)
        string+=".txt"
        current=string
        print("creating feature vector for"+current+"...")
        tuplee=check_contents(dictionary, current)
        feature_train.write(tuplee[1][:-1]+"\n")
    for i in range(1,241):
        string="data_filesv2/tech/"
        if(len(str(i))==1):
            string+="00"+str(i)
        elif(len(str(i))==2):
            string+="0"+str(i)
        else:
            string+=str(i)
        string+=".txt"
        current=string
        print("creating feature vector for"+current+"...")
        tuplee=check_contents(dictionary, current)
        feature_train.write(tuplee[1][:-1]+"\n")

def create_feature_te():
    dictionary=[]
    feature_vector=[]
    file=open("dictionary.txt", "r")
    feature_train=open("dataset-test.csv", "w")
    for i in file.readlines():
        i=i.strip()
        dictionary.append(i)

    for i in range(307,511):
        string="data_filesv2/business/"
        if(len(str(i))==1):
            string+="00"+str(i)
        elif(len(str(i))==2):
            string+="0"+str(i)
        else:
            string+=str(i)
        string+=".txt"
        current=string
        print("creating feature vector for"+current+"...")
        tuplee=check_contents(dictionary, current)
        feature_train.write(tuplee[1][:-1]+"\n")
    for i in range(233,387):
        string="data_filesv2/entertainment/"
        if(len(str(i))==1):
            string+="00"+str(i)
        elif(len(str(i))==2):
            string+="0"+str(i)
        else:
            string+=str(i)
        string+=".txt"
        current=string
        print("creating feature vector for"+current+"...")
        tuplee=check_contents(dictionary, current)
        feature_train.write(tuplee[1][:-1]+"\n")
    for i in range(251,418):
        string="data_filesv2/politics/"
        if(len(str(i))==1):
            string+="00"+str(i)
        elif(len(str(i))==2):
            string+="0"+str(i)
        else:
            string+=str(i)
        string+=".txt"
        current=string
        print("creating feature vector for"+current+"...")
        tuplee=check_contents(dictionary, current)
        feature_train.write(tuplee[1][:-1]+"\n")
    for i in range(308,512):
        string="data_filesv2/sport/"
        if(len(str(i))==1):
            string+="00"+str(i)
        elif(len(str(i))==2):
            string+="0"+str(i)
        else:
            string+=str(i)
        string+=".txt"
        current=string
        print("creating feature vector for"+current+"...")
        tuplee=check_contents(dictionary, current)
        feature_train.write(tuplee[1][:-1]+"\n")
    for i in range(241,400):
        string="data_filesv2/tech/"
        if(len(str(i))==1):
            string+="00"+str(i)
        elif(len(str(i))==2):
            string+="0"+str(i)
        else:
            string+=str(i)
        string+=".txt"
        current=string
        print("creating feature vector for"+current+"...")
        tuplee=check_contents(dictionary, current)
        feature_train.write(tuplee[1][:-1]+"\n")

def multinomial():
    file=open("dataset-training.csv", "r")
    labels=[]
    feature_matrix=[]
    model=MultinomialNB(alpha=0.1)
    d=0
    n=0
    print("Training Data...")
    for i in range(1,307):
        n+=1
        i=file.readline()
        instance=i.split(',')
        try:
            row=list(map(int, instance))
        except:
            row=[]
        labels.append(0)
        feature_matrix.append(row)
    for i in range(1,233):
        n+=1
        i=file.readline()
        instance=i.split(',')
        try:
            row=list(map(int, instance))
        except:
            row=[]
        labels.append(1)
        feature_matrix.append(row)
    for i in range(1,251):
        n+=1
        i=file.readline()
        instance=i.split(',')
        try:
            row=list(map(int, instance))
        except:
            row=[]
        labels.append(2)
        feature_matrix.append(row)
    for i in range(1,308):
        n+=1
        i=file.readline()
        instance=i.split(',')
        try:
            row=list(map(int, instance))
        except:
            row=[]
        labels.append(3)
        feature_matrix.append(row)
    for i in range(1,241):
        n+=1
        i=file.readline()
        instance=i.split(',')
        try:
            row=list(map(int, instance))
        except:
            row=[]
        labels.append(4)
        feature_matrix.append(row)
    model.fit(feature_matrix, np.array(labels))
    file=open("dataset-training.csv", "r")
    prediction=[]
    print("Predicting Training Data...")
    for q in range(1,307):
        i=file.readline()
        instance=i.split(',')
        row=list(map(int,instance))
        k=model.predict((np.array(row)).reshape(-1,len(row)))
        prediction.append(k)
    for q in range(1,233):
        i=file.readline()
        instance=i.split(',')
        row=list(map(int,instance))
        k=model.predict((np.array(row)).reshape(-1,len(row)))
        prediction.append(k)
    for q in range(1,251):
        i=file.readline()
        instance=i.split(',')
        row=list(map(int,instance))
        k=model.predict((np.array(row)).reshape(-1,len(row)))
        prediction.append(k)
    for q in range(1,308):
        i=file.readline()
        instance=i.split(',')
        row=list(map(int,instance))
        k=model.predict((np.array(row)).reshape(-1,len(row)))
        prediction.append(k)
    for q in range(1,241):
        i=file.readline()
        instance=i.split(',')
        row=list(map(int,instance))
        k=model.predict((np.array(row)).reshape(-1,len(row)))
        prediction.append(k)
    print("Multinomial: ", end="")
    print(accuracy_score(labels, prediction))
    print("Predicting Test Data...")
    file=open("dataset-test.csv", "r")
    for q in range(307,511):
        labels.append(0)
        i=file.readline()
        instance=i.split(',')
        row=list(map(int,instance))
        k=model.predict((np.array(row)).reshape(-1,len(row)))
        prediction.append(k)
    for q in range(233,387):
        i=file.readline()
        labels.append(1)
        instance=i.split(',')
        row=list(map(int,instance))
        k=model.predict((np.array(row)).reshape(-1,len(row)))
        prediction.append(k)
    for q in range(251,418):
        labels.append(2)
        i=file.readline()
        instance=i.split(',')
        row=list(map(int,instance))
        k=model.predict((np.array(row)).reshape(-1,len(row)))
        prediction.append(k)
    for q in range(308,512):
        labels.append(3)
        i=file.readline()
        instance=i.split(',')
        row=list(map(int,instance))
        k=model.predict((np.array(row)).reshape(-1,len(row)))
        prediction.append(k)
    for q in range(241,400):
        labels.append(4)
        i=file.readline()
        instance=i.split(',')
        row=list(map(int,instance))
        k=model.predict((np.array(row)).reshape(-1,len(row)))
        prediction.append(k)
    print("Multinomial: ", end="")
    print(accuracy_score(labels[n+1:len(labels)], prediction[n+1:len(labels)]))
    
d=enchant.Dict("EN-US")
# preprocessing()
# create_dictionary(d)
# create_feature_tr()
# create_feature_te()
multinomial()