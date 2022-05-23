"""
A script that performs a classification using a tfidfvectorizer and a logistic regression classifier.
"""

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


def parse_args():
    '''
    Function that specifies the available commandline arguments
    '''
    # Initialise argparse
    ap = argparse.ArgumentParser()
    
    # command line parameters
    ap.add_argument("-f", "--file_input", required = False, default = os.path.join('data', 'VideoCommentsThreatCorpus.csv'), help = "A CSV file", type = int)
    ap.add_argument("-max_df", "--max_df", required = False, default = 0.95, help = "cut-off to remove very common words", type=float)
    ap.add_argument("-min_df", "--min_df", required = False, default = 0.05, help = "cut-off to remove very rare words", type = float)
    ap.add_argument("-max_features", "--max_features", required = False, default = 500, help = "the number of features to keep in the vocabulary ordered by frequency", type = int)
        
    args = vars(ap.parse_args())
    return args


# functions for text processing
def remove_accented_chars(text):
    '''
    Function that removes accented characters.

    text: text in string format
    '''
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def pre_process_corpus(docs):
    '''
    Function that preprocesses text data by removing special characters and lowercasing it.

    docs: text data in an iterable format
    '''
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


def load_data(filepath):
    '''
    Function that loads the data CSV file and splits it into 80% training and 20% testing data.

    filepath: the path to the CSV file
    '''
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
    '''
    Function that initiates a vectorizer and fits it to the training and testing data

    X_train: training data
    X_test: testing data
    '''
    args = parse_args()
    
    MAX_DF = args['max_df']
    MIN_DF = args['min_df']
    MAX_FEATURES = args['max_features']
    
    vectorizer = TfidfVectorizer(ngram_range = (1,2),     # unigrams and bigrams      
                             max_df = MAX_DF,
                             min_df = MIN_DF,
                             max_features = MAX_FEATURES)
    
    # fit to training and test data
    X_train_feats = vectorizer.fit_transform(X_train)
    X_test_feats = vectorizer.transform(X_test)

    return X_train_feats, X_test_feats


def classify(X_train_feats, X_test_feats, y_train, y_test):
    '''
    Function that creates and trains the logistic regression classifier. It then predicts the results on the testing data. A classification report is also created and saved. 
    
    X_train_feats: the training data after vectorization
    X_test_feats: the testing data after vectorization
    y_train: the training data labels
    y_test: the testing data labels
    '''
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


def main():
    '''
    The process of the script.
    '''
    args = parse_args()
    file_input = args['file_input']
    file_path = os.path.join(file_input)
    
    print('[INFO] Loading data')
    X_train, X_test, y_train, y_test = load_data(file_input)

    print('[INFO] Initiating vectorizer ...')
    X_train_feats, X_test_feats = initiate_vectorizer(X_train, X_test)
    classify(X_train_feats, X_test_feats, y_train, y_test)

    print('[INFO] script success')

    return
                

if __name__ == '__main__':
    main()