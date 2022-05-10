import string
from typing import Optional
from LRModel import ModelL
from SVMModel import  ModelS
from pydantic import BaseModel


import pandas as pd
import numpy as np
import sys
import nltk

import re, string, unicodedata
import contractions
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet




from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, plot_precision_recall_curve, plot_confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin

import matplotlib.pyplot as plt

from collections import Counter

import seaborn as sns

import gensim
from gensim.models import Word2Vec


from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve
from statistics import mode

from pandas.core.dtypes.generic import ABCIndex
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

from fastapi import FastAPI
nltk.download('omw-1.4')
from fastapi.middleware.cors import CORSMiddleware

class DataModel(BaseModel):
    study_and_condition : str

    def columns(self):
        return ["study_and_condition"]


app = FastAPI()

origins = ['*']

app.add_middleware(CORSMiddleware, allow_origins = origins, allow_credentials = True, allow_headers = ['*'], allow_methods = ['*'])

@app.get("/")
def read_root():
   return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}

@app.get("/SVM/scores")
def svmScores():
    model = ModelS()

    return {"Exactitud": model.getExactitud()}


@app.get("/LR/scores")
def lrScores():
    model = ModelL()

    return {"Exactitud": model.getExactitud()}


@app.post("/SVM")
def predSVM(dataModel: DataModel):
    data = dataModel.study_and_condition
    dataCondition = data.split('.', 1)[1]
    expanded =[]
    for word in dataCondition.split():
        expanded.append(contractions.fix(word))
    expanded_text = ' '.join(expanded)
    dataConditionTok = word_tokenize(expanded_text)
    dataCondition = preprocessing(dataConditionTok)
    dataCondition = finalpreprocess(dataCondition)
    dataConditionTok = word_tokenize(dataCondition)
    arr =[]
    arr.append(dataCondition)
    df =pd.DataFrame(arr, columns=['condition'])
    
    model = ModelS()
    print(df['condition'])
    x = model.make_predictions(df['condition'])
    print(x)
    xP = model.make_predictions_probability(df['condition'])
    print(xP)
    return {"clasificacion": x, "prob0": xP[0][0] , "prob1": xP[0][1]}
  


@app.post("/LR")
def predSVM(dataModel: DataModel):
    data = dataModel.study_and_condition
    dataCondition = data.split('.', 1)[1]
    expanded =[]
    for word in dataCondition.split():
        expanded.append(contractions.fix(word))
    expanded_text = ' '.join(expanded)
    dataConditionTok = word_tokenize(expanded_text)
    dataCondition = preprocessing(dataConditionTok)
    dataCondition = finalpreprocess(dataCondition)
    dataConditionTok = word_tokenize(dataCondition)
    arr =[]
    arr.append(dataCondition)
    df =pd.DataFrame(arr, columns=['condition'])
    
    model = ModelL()
    print(df['condition'])
    x = model.make_predictions(df['condition'])
    print(x)
    xP = model.make_predictions_probability(df['condition'])
    print(xP)
    return {"clasificacion": x, "prob0": xP[0][0] , "prob1": xP[0][1]}

@app.post("/mixed")
def predBoth(dataModel :DataModel):
    data = dataModel.study_and_condition
    dataCondition = data.split('.', 1)[1]
    expanded =[]
    for word in dataCondition.split():
        expanded.append(contractions.fix(word))
    expanded_text = ' '.join(expanded)
    dataConditionTok = word_tokenize(expanded_text)
    dataCondition = preprocessing(dataConditionTok)
    dataCondition = finalpreprocess(dataCondition)
    dataConditionTok = word_tokenize(dataCondition)
    arr =[]
    arr.append(dataCondition)
    df =pd.DataFrame(arr, columns=['condition'])
    
    model = ModelS()
    print(df['condition'])
    x = model.make_predictions(df['condition'])
    print(x)
    pfinal = 0
    prob = 0
    prob2 = 0
    result = '__label__0'

    if x == '__label__1':
        xP = model.make_predictions_probability(df['condition'])
        prob = xP[0][1]
        model = ModelL()
        x = model.make_predictions(df['condition'])
        xP =  model.make_predictions_probability(df['condition'])
        prob2 = xP[0][1]
        if x == '__label__1':
            result = '__label__1'
            pfinal = prob*prob2
        else :
            result = 'undet'

    
   
    
    return {"clasificacion": result, "prob":pfinal}
        
    
    


    
    

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower();
        new_words.append(new_word);
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def stopword(words):
    """Remove stop words from list of tokenized words"""
    a= [i for i in words.split() if i not in stopwords.words('english')]
    return ' '.join(a)

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    stop_words = set(stopwords.words('english'))
    for word in words:
        if word not in stop_words:
            new_word = word
            new_words.append(new_word)
    return new_words

def preprocessing(words):
    words = to_lowercase(words)
    words = replace_numbers(words)
    words = remove_punctuation(words)
    words = remove_non_ascii(words)
    words = remove_stopwords(words)
    textoFinal = ""
    for x in words:
        textoFinal+=x+" "
    return textoFinal

wl = WordNetLemmatizer()

def preprocess(words):
    words = words.lower() 
    words= words.strip()  
    words=re.compile('<.*?>').sub('', words) 
    words = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', words)  
    words = re.sub('\s+', ' ', words)  
    words = re.sub(r'\[[0-9]*\]',' ',words) 
    words=re.sub(r'[^\w\s]', '', str(words).lower().strip())
    words = re.sub(r'\d',' ',words) 
    words = re.sub(r'\s+',' ',words) 
    return words

# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
# Tokenize the sentence
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)

def finalpreprocess(string):
    return lemmatizer(preprocess(string))    
    
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
