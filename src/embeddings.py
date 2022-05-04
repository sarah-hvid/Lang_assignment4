"""
A script that performs a classification using embeddings and a CNN.
As input, provide the data in the data folder. 

Usage example:
    $ python src/embeddings.py
    
    With all specifications:
    $ python embeddings.py -epochs 2 -batch_size 128 -embed_size 300
    
    
Requirements:
The folder structure must be the same as in the GitHub repository.
The current working directory when running the script must be the one that contains the data, output and src folder. 
The output folder must be named output.

"""

#----------------------------- importing packages --------------------------------------

# system tools
import argparse
import os

# text processing tools
import re
import unicodedata
import contractions

# data wrangling
import pandas as pd
import numpy as np

# tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, 
                                    Flatten,
                                    Conv1D, 
                                    MaxPooling1D, 
                                    Embedding)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

# scikit-learn
from sklearn.metrics import (confusion_matrix, 
                            classification_report)
from sklearn.model_selection import train_test_split


#----------------------------- creating functions --------------------------------------

# argument function
def parse_args():
    # Initialise argparse
    ap = argparse.ArgumentParser()
    
    # command line parameters
    ap.add_argument("-epochs", "--epoch_num", required = False, help = "number of epochs", type = int)
    ap.add_argument("-batch_size", "--batch_size", required = False, help = "the batch size", type = int)
    ap.add_argument("-embed_size", "--embed_size", required = False, help = "the number of dimensions for embeddings", type = int)
        
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


# functions for data preprocessing
def load_data():
    filepath = os.path.join('data', 'VideoCommentsThreatCorpus.csv')
    data = pd.read_csv(filepath)
    
    labels = data['label'].values # keeping the data as a numpy array
    texts = data['text'].values
    
    X_train, X_test, y_train, y_test = train_test_split(texts, 
                                                            labels,
                                                            train_size= round(len(texts) * 0.8), 
                                                            test_size=round(len(texts) * 0.2))
    return X_train, X_test, y_train, y_test



def preprocess_data(X_train, X_test):
    
    MAX_SEQUENCE_LENGTH = 1000
    
    X_train_norm = pre_process_corpus(X_train)
    X_test_norm = pre_process_corpus(X_test)
    
    # define out-of-vocabulary token
    t = Tokenizer(oov_token = '<UNK>')

    #fit the tokenizer on the documents
    t.fit_on_texts(X_train_norm)

    # add padding value
    t.word_index['<PAD>'] = 0
    
    X_train_seqs = t.texts_to_sequences(X_train_norm)
    X_test_seqs = t.texts_to_sequences(X_test_norm)
    
    # add padding to sequences
    X_train_pad = sequence.pad_sequences(X_train_seqs, maxlen = MAX_SEQUENCE_LENGTH)
    X_test_pad = sequence.pad_sequences(X_test_seqs, maxlen = MAX_SEQUENCE_LENGTH)
    
    return t, X_train_pad, X_test_pad


# functions for model building
def compile_model(t, X_train_pad, y_train):
        
    MAX_SEQUENCE_LENGTH = 1000
    # overall vocabulary size
    VOCAB_SIZE = len(t.word_index)
    
    args = parse_args()
    
    EPOCHS = args['epoch_num']
    BATCH_SIZE = args['batch_size']
    EMBED_SIZE = args['embed_size']
    
    if EPOCHS == None:
        EPOCHS = 1
    else:
        pass
    
    if BATCH_SIZE == None:
        BATCH_SIZE = 128
    else:
        pass
    
    if EMBED_SIZE == None:
        EMBED_SIZE = 300
    else:
        pass


    # create the model
    model = Sequential()
    # embedding layer
    model.add(Embedding(VOCAB_SIZE, 
                        EMBED_SIZE, 
                        input_length=MAX_SEQUENCE_LENGTH))

    # first convolution layer and pooling
    model.add(Conv1D(filters=128, 
                            kernel_size=4, 
                            padding='same',
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # second convolution layer and pooling
    model.add(Conv1D(filters=64, 
                            kernel_size=4, 
                            padding='same', 
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # third convolution layer and pooling
    model.add(Conv1D(filters=32, 
                            kernel_size=4, 
                            padding='same', 
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    # fully-connected classification layer
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', 
                            optimizer='adam', 
                            metrics=['accuracy'])

    
    h = model.fit(X_train_pad, y_train,
              epochs = EPOCHS,
              batch_size = BATCH_SIZE,
              validation_split = 0.1,
              verbose = 2)  # don't print each epoch
    
    return model


def evaluate_model(model, X_test_pad, y_test):
    
    labels = ['non-toxic', 'toxic']
    
    # 0.5 decision boundary
    predictions = (model.predict(X_test_pad) > 0.5).astype("int32")
    
    report = classification_report(y_test, predictions)
    print(report)
    
    df = pd.DataFrame(confusion_matrix(y_test, predictions), 
                 index=labels, columns=labels)
    print(df)
    
    with open("output/embeddings_report.txt", "w") as f:
            print(report, file=f)
    
    return 
    
#----------------------------- process of the script --------------------------------------

def main():
    
    X_train, X_test, y_train, y_test = load_data()
    t, X_train, X_test = preprocess_data(X_train, X_test)
    model = compile_model(t, X_train, y_train)
    evaluate_model(model, X_test, y_test)
    
    return
                

if __name__ == '__main__':
    main()