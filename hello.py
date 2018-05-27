print("Importing...")
import re
import enchant
import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from nltk.stem import *
from time import time

import os
from os import path

# Naive Bayes models (list of tuples containing model, and string for printing)
nbmodels = [
    (BernoulliNB(), 'Bernoulli NB with default'),
    (GaussianNB(), 'Gaussian NB with default'),
    #(MultinomialNB(alpha=0.3), 'Multinomial NB with alpha=0.3'),
    (MultinomialNB(alpha=0.1), 'Multinomial NB with alpha=0.1'),
    #(MultinomialNB(alpha=0.03), 'Multinomial NB with alpha=0.03'),
    #(MultinomialNB(alpha=0.01), 'Multinomial NB with alpha=0.01'),
]
# Non-NB models (using standardized data)
models = [
    (SVC(), 'SVM with default'),
    (SVC(kernel='poly', degree=1), 'Polynomial SVM with degree 1'),
    #(SVC(kernel='poly', degree=2), 'Polynomial SVM with degree 2'),
]

def create_clean(news):
    unwantedPatterns = [r'\'s', r'[0-9]+', r'[\%\(\)\.\,\;\:\"\-]', r'\s+']
    raw=open(path.join('data_files', news),"r", encoding="ISO-8859-1")
    cleaned=open(path.join('data_filesv2', news),"w")
    processed=raw.read()
    processed=processed.lower()
    for x in unwantedPatterns:
        processed=re.sub(x, r' ', processed)
    cleaned.write(processed)

def preprocessing():
    print("Preprocessing...")
    # list all classes (business, tech, etc.) found in data_files
    for newsClass in os.listdir('data_files'):
        # parse file found in each class
        for filename in os.listdir(path.join('data_files', newsClass)):
            create_clean(path.join(newsClass, filename))

def create_dictionary(d):
    print("Creating dictionary...")
    dictionary=set()
    for newsClass in os.listdir('data_filesv2'):
        for filename in os.listdir(path.join('data_filesv2', newsClass)):
            raw=open(path.join('data_filesv2', newsClass, filename), "r")
            text=raw.read()
            words=text.split()
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
    dictfile = open("dictionary.txt", "w")
    wordlist=list(dictionary)
    wordlist.sort()
    for i in wordlist:
       dictfile.write(i+"\n")

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

def create_feature_csv(csvfile, lowerPercent, upperPercent):
    print("Creating csv file: %s..." % csvfile)
    dictionary=[]
    feature_vector=[]
    file=open("dictionary.txt", "r")
    feature_train=open(csvfile, "w")
    for i in file.readlines():
        i=i.strip()
        dictionary.append(i)

    for newsClass in sorted(os.listdir('data_filesv2')):
        sortedFiles = sorted(os.listdir(path.join('data_filesv2', newsClass)))
        lowerBound = int(len(sortedFiles) * lowerPercent)
        upperBound = int(len(sortedFiles) * upperPercent)
        for filename in sortedFiles[lowerBound:upperBound]:
            current = path.join('data_filesv2', newsClass, filename)
            print("creating feature vector for"+current+"...")
            tuplee=check_contents(dictionary, current)
            feature_train.write(tuplee[1][:-1]+"\n")

def fit_predict_show(modelTuple, XTrain, YTrain, XTest, YTest):
    model = modelTuple[0]
    description = modelTuple[1]
    print('"' + description + '"')
    startTime = time()
    model.fit(XTrain, YTrain)
    print("\tTraining time:   %.3fs" % (time() - startTime))
    startTime = time()
    PTrain = model.predict(XTrain)
    PTest = model.predict(XTest)
    print("\tPredicting time: %.3fs" % (time() - startTime))
    print("\tTraining accuracy: %.3f%%" % (accuracy_score(YTrain, PTrain) * 100))
    print("\tTest accuracy:     %.3f%%" % (accuracy_score(YTest, PTest) * 100))

def multinomial():
    classesLen = {}
    for newsClass in sorted(os.listdir('data_files')):
        classesLen[newsClass] = len(os.listdir(path.join('data_files', newsClass)))
    trainfile=open("dataset-training.csv", "r")
    testfile=open("dataset-test.csv", "r")
    XTrain=[]
    YTrain=[]
    XTest=[]
    YTest=[]
    print("Reading training set")
    for newsClass in classesLen:
        lowerBound = int(classesLen[newsClass] * 0)
        upperBound = int(classesLen[newsClass] * 6/10)
        for i in range(lowerBound,upperBound):
            csvline=trainfile.readline()
            instance=csvline.split(',')
            try:
                row=list(map(int, instance))
            except:
                row=[]
            YTrain.append(newsClass)
            XTrain.append(row)
    print("Reading test set")
    for newsClass in classesLen:
        lowerBound = int(classesLen[newsClass] * 6/10)
        upperBound = int(classesLen[newsClass] * 1)
        for i in range(lowerBound,upperBound):
            csvline=testfile.readline()
            instance=csvline.split(',')
            try:
                row=list(map(int, instance))
            except:
                row=[]
            YTest.append(newsClass)
            XTest.append(row)
    
    # Fit, predict, and show accuracies of each model on training and test sets
    print("~~NAIVE BAYES CLASSIFIERS~~")
    for model in nbmodels:
        fit_predict_show(model, XTrain, YTrain, XTest, YTest)

    # Standardize feature vectors
    scaler = StandardScaler()
    scaler.fit(XTrain)
    XTrain = scaler.transform(XTrain)
    XTest = scaler.transform(XTest)
    # Fit, predict, and show accuracies of each model on training and test sets
    print("~~OTHER CLASSIFIERS~~")
    for model in models:
        fit_predict_show(model, XTrain, YTrain, XTest, YTest)

#preprocessing()
#create_dictionary(enchant.Dict("EN-US"))
#create_feature_csv('dataset-training.csv', 0, 0.6)
#create_feature_csv('dataset-test.csv', 0.6, 1)
multinomial()
