# Assignment 4 - Text classification
 
#### Link to GitHub of this assignment: https://github.com/sarah-hvid/Lang_assignment4
 
## Assignment description
In this assignment it should be predicted whether textual data is toxic or non-toxic, using two different methods:
- Machine learning approach using a ```TfidfVectorizer()``` and a ```LogisticRegression()``` classifier
- Deep learning approach using a Keras ```Embedding``` layer and a Convolutional Neural Network

The full assignment description is available in the ```assignment4.md``` file.

## Methods
This problem relates to classification of text. Initially, the data was split into a training and test set containing 80% and 20% of the data, respectively. Then, the data was cleaned by removing odd characters and lowercasing it. Here, the process differs for the two different methods:

- The machine learning approach:
  - The tfidf-vectorizer is initiated with unigrams and bigrams. The maximum and minimum document frequency terms to be kept in the vocabulary may be specified by the user. The maximum number of features to be kept may also be specified by the user. The vectorizer is then applied to the data. The vectorized data is then classified using logistic regression. The classification report is saved in the output folder. 
  
- The deep learning approach:
  - A tokenizer is created and fitted to the data. The CNN model is then created. The number of epochs, the batch size and the embedding dimension size of the embedding layer may be specified by the user. The model was then fitted to the data. The classification report is saved in the output folder. 

## Usage
In order to run the scripts, certain modules need to be installed. These can be found in the ```requirements.txt``` file. The folder structure must be the same as in the GitHub repository. The current working directory when running the script must be the one that contains the ```data```, ```output``` and ```src``` folder. Examples of how to run the scripts from the command line: 

The tfidf-vectorizer script:
- python src/tfidfvectorizer.py
- python src/tfidfvectorizer.py -max_df 0.8 -min_df 0.1 -max_features 500

The embeddings script:
- python src/embeddings.py
- python transfer_learning.py -epochs 2 -batch_size 128 -embed_size 300
  
Examples of the outputs of the scripts may be seen in the ```output``` folder. 

## Results
The machine learning approach is clearly struggling with this data. It initially seems like the classifier is doing well, but upon further inspection it can be seen that the model simply classifies everything, or almost everything, as non-toxic. This may be related to the small amount of toxic data available for training. The deep learning approach achieved better results, but still only an accuracy (f1-score) of 0.6 - 0.7. Overall, I believe more data is required to improve the accuracy for both approaches. 
