print("Importing...")
import re
import enchant
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from nltk.stem import *
from time import time

import os
from os import path

# Naive Bayes models (list of tuples containing model, description)
nbmodels = [
    (BernoulliNB(), 'Bernoulli NB with default'),
    (GaussianNB(), 'Gaussian NB with default'),
    (MultinomialNB(alpha=0.3), 'Multinomial NB with alpha=0.3'),
    (MultinomialNB(alpha=0.1), 'Multinomial NB with alpha=0.1'),
    (MultinomialNB(alpha=0.03), 'Multinomial NB with alpha=0.03'),
    (MultinomialNB(alpha=0.01), 'Multinomial NB with alpha=0.01'),
]
# Non-NB models (using standardized data)
svmmodels = [
    (SVC(), 'SVM with default'),
    (SVC(kernel='poly', degree=1), 'Polynomial SVM with degree 1'),
    (SVC(kernel='poly', degree=2), 'Polynomial SVM with degree 2'),
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

def create_sparse_matrix(description, lowerPercent, upperPercent):
    print("Generating matrix: %s..." % description)
    dictionary=[]
    file=open("dictionary.txt", "r")
    for i in file.readlines():
        i=i.strip()
        dictionary.append(i)
    dictIndex = {dictionary[i]: i for i in range(len(dictionary))}
    rowIndex = []
    colIndex = []
    data = []

    Y = []
    i = 0
    for newsClass in sorted(os.listdir('data_filesv2')):
        sortedFiles = sorted(os.listdir(path.join('data_filesv2', newsClass)))
        lowerBound = int(len(sortedFiles) * lowerPercent)
        upperBound = int(len(sortedFiles) * upperPercent)
        for filename in sortedFiles[lowerBound:upperBound]:
            current = path.join('data_filesv2', newsClass, filename)
            words = open(current, 'r').read().split()
            for word in words:
                j = dictIndex.get(word)
                if j:
                    rowIndex.append(i)
                    colIndex.append(j)
                    data.append(1)
            Y.append(newsClass)
            i += 1
    X = coo_matrix((data, (rowIndex, colIndex)), shape=(i, len(dictionary)))
    return (X, Y)

def fit_predict_show(modelTuple, XTrain, YTrain, XTest, YTest):
    model = modelTuple[0]
    description = modelTuple[1]
    print('"' + description + '"')
    startTime = time()
    try:
        model.fit(XTrain, YTrain)
    except TypeError:
        XTrain = XTrain.toarray()
        XTest = XTest.toarray()
        model.fit(XTrain, YTrain)
    print("\tTraining time:   %.3fs" % (time() - startTime))
    startTime = time()
    PTrain = model.predict(XTrain)
    PTest = model.predict(XTest)
    print("\tPredicting time: %.3fs" % (time() - startTime))
    print("\tTraining accuracy: %.3f%%" % (accuracy_score(YTrain, PTrain) * 100))
    print("\tTest accuracy:     %.3f%%" % (accuracy_score(YTest, PTest) * 100))

def multinomial(XTrain, YTrain, XTest, YTest):
    # Fit, predict, and show accuracies of each model on training and test sets
    print("~~NAIVE BAYES CLASSIFIERS~~")
    for modelTuple in nbmodels:
        fit_predict_show(modelTuple, XTrain, YTrain, XTest, YTest)

    XTrain = XTrain.toarray().astype(np.float64)
    XTest = XTest.toarray().astype(np.float64)
    # Standardize feature vectors
    scaler = StandardScaler()
    scaler.fit(XTrain)
    XTrain = scaler.transform(XTrain)
    XTest = scaler.transform(XTest)
    # Fit, predict, and show accuracies of each model on training and test sets
    print("~~OTHER CLASSIFIERS~~")
    for modelTuple in svmmodels:
        fit_predict_show(modelTuple, XTrain, YTrain, XTest, YTest)

#preprocessing()
#create_dictionary(enchant.Dict("EN-US"))
(XTrain, YTrain) = create_sparse_matrix("Train", 0, 0.6)
(XTest, YTest) = create_sparse_matrix("Test", 0.6, 1)
multinomial(XTrain, YTrain, XTest, YTest)
