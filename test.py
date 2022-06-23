from random import random
from xml.parsers.expat import model
from flask import Flask,render_template,url_for,request
from flask import Flask, render_template, request, redirect
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import numpy as np
import nltk
from sklearn.datasets import load_files
# nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
from sklearn.svm import SVC
from nltk.tokenize import word_tokenize
import string 
from nltk.stem import WordNetLemmatizer
import sqlite3
import speech_recognition as sr

m_cols = ('class','text')
df = pd.read_csv("all-data.csv", names=m_cols , encoding='latin-1')

#Cleaning data

Tweet = []
Labels = []

for row in df["text"]:
    #tokenize words
    words = word_tokenize(row)
    #remove punctuations
    clean_words = [word.lower() for word in words if word not in set(string.punctuation)]
    #remove stop words
    english_stops = set(stopwords.words('english'))
    characters_to_remove = ["''",'``',"rt","https","’","“","”","\u200b","--","n't","'s","...","//t.c" ]
    clean_words = [word for word in clean_words if word not in english_stops]
    clean_words = [word for word in clean_words if word not in set(characters_to_remove)]
    #Lematise words
    wordnet_lemmatizer = WordNetLemmatizer()
    lemma_list = [wordnet_lemmatizer.lemmatize(word) for word in clean_words]
    Tweet.append(lemma_list)



df['message']=df['text']

from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
df['class']= label_encoder.fit_transform(df['class'])
df['class'].unique()

df = df[0:1000]
X = df['message']
y = df['class']

# Extract Feature With CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X) # Fit the Data



with open('./model.pkl', 'rb') as f: model = pickle.load(f)

data = ["With the new production plant the company would increase its capacity to meet the expected increase in demand and would improve the use of raw materials and therefore increase the production profitability ."]


data = ['The international electronic industry company Elcoteq has laid off tens of employees from its Tallinn facility ; contrary to earlier layoffs the company contracted the ranks of its office workers , the daily Postimees reported .']


data = ["A corresponding increase of 85,432.50 euros in Ahlstrom 's share capital has been entered in the Trade Register today ."]


data = ["Thanks to the internet , consumers compare products more than previously and Finnish companies are not competitive ."]


data = ["The company will also be compensated for acting as a reserve batch plant ."]
vect = cv.transform(data).toarray()

print(model.predict(vect))