import pandas as pd
import nltk
from nltk import stem
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt
import re
import tensorflow as tf
import pickle
from sklearn.utils import shuffle
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM, Bidirectional, Conv1D, GRU
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
from sklearn.metrics import f1_score, precision_score, recall_score,confusion_matrix

class MySpamDetector : 
    def __init__(self):
        self.maxlen = 100
    
    def load_dataset(self,path):
        data = pd.read_csv(path,encoding='latin-1')
        # selecting rows based on condition  
        dataset = data[['text','spam']]
        dataset = shuffle(dataset)
        return dataset
    
    def preprocess_text(self,sentence):
        return sentence

    def get_embedding_matrix(self):
        embeddings_dictionary = dict()
        with open('dataset/glove.6B.100d.txt', encoding="utf8") as glove_file:
            for line in glove_file:
                records = line.split()
                word = records[0]
                vector_dimensions = np.asarray(records[1:], dtype='float32')
                embeddings_dictionary [word] = vector_dimensions

        embedding_matrix = np.zeros((self.vocab_size, 100))
        for word, index in self.tokenizer.word_index.items():
            embedding_vector = embeddings_dictionary.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
        return embedding_matrix

    def get_model(self):
        model = Sequential([
            Embedding(self.vocab_size, 100, weights=[self.embedding_matrix], input_length=self.maxlen , trainable=False),
            Bidirectional(GRU(300,return_sequences=False)),
            Dense(1, activation='sigmoid')])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def get_tokenizer(self,data_train):
        tokenizer = Tokenizer(num_words=10000, oov_token=0)
        tokenizer.fit_on_texts(data_train)
        with open('modules/model/tokenizer_spam.pickle','wb') as handle : 
            pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)
        return tokenizer
    
    def save_model(self,model,filename):
        model.save(filename)

    def load_model(self,filename):
        self.model = tf.keras.models.load_model(filename)
        self.tokenizer = self.load_tokenizer()

    def load_tokenizer(self):
        with open('modules/model/tokenizer_spam.pickle', 'rb') as handle:
            Tokenizer = pickle.load(handle)
        return Tokenizer
    
    def preprocessing(self, sentence):
        stemmer = stem.SnowballStemmer('english')
        stopwords = set(stopwords.words('english'))

        # Lowercase
        sentence = sentence.lower()


    def train(self):
        print('-------- Initiate Trainig --------')
        dataset = self.load_dataset("dataset/dataset_youtube_spam.csv")
        
        X = []
        sentence = list(dataset['text'])
        for sen in sentence:
            X.append(self.preprocess_text(sen))
        y = dataset['spam']

        # Split train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        # Tokenize sentencs to numbers with max number 10000
        self.tokenizer = self.get_tokenizer(X_train)
        X_train = self.tokenizer.texts_to_sequences(X_train)
        X_test = self.tokenizer.texts_to_sequences(X_test)

        # Adding 1 because of reserved 0 index
        self.vocab_size = len(self.tokenizer.word_index) + 1
        X_train = pad_sequences(X_train, padding='post', maxlen=self.maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=self.maxlen)
        self.embedding_matrix = self.get_embedding_matrix()

        # Get model
        self.model = self.get_model()
        print(self.model.summary())
        
        # Training
        self.model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

        # Evaluation using test data
        y_pred = (self.model.predict(X_test) > 0.5).astype("int32")
        self.print_evaluation("SpamDetectorModel", y_test, y_pred)
        self.save_model(self.model,"modules/model/SpamDetectorModel")
    
    def print_evaluation(self, task_name, y_true, y_pred):
        print(task_name)
        print("Precision : ", precision_score(y_true, y_pred, average='macro'))
        print("Recall : ", recall_score(y_true, y_pred, average='macro'))
        print("F1-score : ", f1_score(y_true, y_pred, average='macro'))
        print("Confusion matrix : \n", confusion_matrix(y_true, y_pred))

    def predict(self, sentence, decode=True):
        if isinstance(sentence,str):
            sentence = [sentence]
        sentence = self.tokenizer.texts_to_sequences([sentence])
        sentence = pad_sequences(sentence, padding='post', maxlen=self.maxlen)
        y_pred =  (self.model.predict(sentence) > 0.5).astype("int32")
        print(y_pred)
        if decode:
            y_pred = self.decode(y_pred)
        print(y_pred)
        return y_pred
    
    def decode(self, label):
        if label.all() == 0:
            return 'ham'
        else:
            return 'spam'
