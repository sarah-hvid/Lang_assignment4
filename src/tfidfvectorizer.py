"""
A script that performs a classification using a tfidfvectorizer and a logistic regression classifier.
As input, provide the data in the data folder. 

Usage example:
    $ python src/tfidfvectorizer.py

"""
#----------------------------- importing packages --------------------------------------

# system tools
import os
import argparse

# data tools
import pandas as pd

# text processing tools
import unicodedata
import contractions
import re

# Machine learning stuff
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import (confusion_matrix, 
                            classification_report)

#----------------------------- creating functions --------------------------------------
# argument function
def parse_args():
    # Initialise argparse
    ap = argparse.ArgumentParser()
    
    # command line parameters
    ap.add_argument("-max_df", "--max_df", required = False, help = "cut-off to remove very common words", type=float)
    ap.add_argument("-min_df", "--min_df", required = False, help = "cut-off to remove very rare words", type = float)
    ap.add_argument("-max_features", "--max_features", required = False, help = "the number of features to keep in the vocabulary ordered by frequency", type = int)
        
    args = vars(ap.parse_args())
    return args


# functions for text processing
def remove_accented_chars(text):
  text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
  return text

def pre_process_corpus(docs):
  norm_docs = [] 
  for doc in docs:
    doc = doc.translate(doc.maketrans("\n\t\r", "   "))
    doc = doc.lower()
    doc = remove_accented_chars(doc)
    doc = contractions.fix(doc)
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = re.sub(' +', ' ', doc)
    doc = doc.strip()  
    norm_docs.append(doc)
  
  return norm_docs


def load_data():
    filepath = os.path.join('data', 'VideoCommentsThreatCorpus.csv')
    data = pd.read_csv(filepath)
    
    labels = data['label'].values # keeping the data as a numpy array
    texts = data['text'].values
    
    X_train, X_test, y_train, y_test = train_test_split(texts, 
                                                            labels,
                                                            train_size= round(len(texts) * 0.8), 
                                                            test_size=round(len(texts) * 0.2))
    
    # applying preproccessing function
    X_train = pre_process_corpus(X_train)
    X_test = pre_process_corpus(X_test)
    
    return X_train, X_test, y_train, y_test


def initiate_vectorizer(X_train, X_test):
    
    args = parse_args()
    
    MAX_DF = args['max_df']
    MIN_DF = args['min_df']
    MAX_FEATURES = args['max_features']
    

    if MAX_DF == None:
        MAX_DF = 0.95
    else:
        pass
    
    if MIN_DF == None:
        MIN_DF = 0.05
    else:
        pass
    
    if MAX_FEATURES == None:
        MAX_FEATURES = 500
    else:
        pass
    
    vectorizer = TfidfVectorizer(ngram_range = (1,2),     # unigrams and bigrams      
                             max_df = MAX_DF,
                             min_df = MIN_DF,
                             max_features = MAX_FEATURES)
    
    # fit to training and test data
    X_train_feats = vectorizer.fit_transform(X_train)
    X_test_feats = vectorizer.transform(X_test)

    return X_train_feats, X_test_feats


def classify(X_train_feats, X_test_feats, y_train, y_test):
    
    classifier = LogisticRegression().fit(X_train_feats, y_train)
    
    y_pred = classifier.predict(X_test_feats)
    
    report = metrics.classification_report(y_test, y_pred)
    print(report)
    
    with open("output/tfidf_report.txt", "w") as f:
        print(report, file=f)
        
    
    labels = ['non-toxic', 'toxic']
    df = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                 index=labels, columns=labels)
    print(df)
    
    
    return


#----------------------------- process of the script --------------------------------------

def main():
    
    X_train, X_test, y_train, y_test = load_data()
    X_train_feats, X_test_feats = initiate_vectorizer(X_train, X_test)
    classify(X_train_feats, X_test_feats, y_train, y_test)

    return
                

if __name__ == '__main__':
    main()