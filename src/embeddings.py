"""
A script that performs a classification using embeddings and a CNN.
"""

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


# argument function
def parse_args():
    '''
    Function that specifies the available commandline arguments
    '''
    # Initialise argparse
    ap = argparse.ArgumentParser()
    
    # command line parameters
    ap.add_argument("-f", "--file_input", required = False, default = os.path.join('data', 'VideoCommentsThreatCorpus.csv'), help = "A CSV file")
    ap.add_argument("-epochs", "--epoch_num", required = False, default = 5, help = "number of epochs", type = int)
    ap.add_argument("-batch_size", "--batch_size", required = False, default = 128, help = "the batch size", type = int)
    ap.add_argument("-embed_size", "--embed_size", required = False, default = 300, help = "the number of dimensions for embeddings", type = int)
        
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


def preprocess_corpus(docs):
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


# functions for data preprocessing
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
    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test):
    '''
    Function that preprocesses the training data by creating and fitting a tokenizer.

    X_train: the training data text as a numpy array
    X_test: the testing data text as a numpy array
    '''
    MAX_SEQUENCE_LENGTH = 1000
    
    X_train_norm = preprocess_corpus(X_train)
    X_test_norm = preprocess_corpus(X_test)
    
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
    '''
    Function that compiles a CNN model with an embedding layer and trains it on training data.

    t: tokenizer that has been fitted to the data
    X_train_pad: preprocessed training data
    y_train: training data labels
    '''
        
    MAX_SEQUENCE_LENGTH = 1000
    # overall vocabulary size
    VOCAB_SIZE = len(t.word_index)
    
    args = parse_args()
    
    EPOCHS = args['epoch_num']
    BATCH_SIZE = args['batch_size']
    EMBED_SIZE = args['embed_size']

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

    # fit the model to the data
    h = model.fit(X_train_pad, y_train,
              epochs = EPOCHS,
              batch_size = BATCH_SIZE,
              validation_split = 0.1,
              verbose = 2)
    
    return model


def evaluate_model(model, X_test_pad, y_test):
    '''
    Function that evaluates the model performance on the test data. It also prints and saves the results.

    model: a tensorflow model object.
    X_test_pad: preprocessed testing data
    y_test: testing data labels
    '''
    args = parse_args()
    
    EPOCHS = args['epoch_num']
    BATCH_SIZE = args['batch_size']
    EMBED_SIZE = args['embed_size']
    
    labels = ['non-toxic', 'toxic']

    # creating predictions of the model
    predictions = (model.predict(X_test_pad) > 0.5).astype("int32")
    
    # creating classification report and saving the results
    report = classification_report(y_test, predictions, target_names = labels)
    print(report)
    
    with open(f"output/embeddings_{EPOCHS}_{BATCH_SIZE}_{EMBED_SIZE}_report.txt", "w") as f:
            print(report, file=f)
    
    return 
    

def main():
    '''
    The process of the script.
    '''
    args = parse_args()
    file_input = args['file_input']
    file_path = os.path.join(file_input)

    print('[INFO] Loading data')
    X_train, X_test, y_train, y_test = load_data(file_path)
    t, X_train, X_test = preprocess_data(X_train, X_test)
    print('[INFO] compilling model ...')
    model = compile_model(t, X_train, y_train)
    evaluate_model(model, X_test, y_test)

    print('[INFO] script success')

    return
                

if __name__ == '__main__':
    main()