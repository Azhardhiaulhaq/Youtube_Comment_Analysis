import re
import pandas as pad_sequences
import nltk
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score,confusion_matrix

class SpamModules(object):
    def __init__(self,config):
        self.config = config
        self.model = None
    
    def init_model(self, word_embedding):
        self.model = Sequential([
            layers.Embedding(word_embedding.vocab_size, word_embedding.dim, weights=[word_embedding.embeddings_matrix], input_length=self.config.max_len, trainable=False),
            layers.Bidirectional(layers.GRU(self.config.spam_config.hidden_size,return_sequences=False)),
            layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
    def train(self,X_train,y_train):
        print(X_train.shape)
        
        history = self.model.fit(X_train, y_train, validation_split=0.2, epochs=self.config.spam_config.epochs, batch_size=self.config.spam_config.batch_size)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('spam_loss')
    
    def predict_one(self,X,tokenizer=None, decode=False):
        if tokenizer is not None:
            X = tokenizer(X)
        scores = self.model.predict_classes(X)
        print(scores)
        if decode:
            return self.decode(scores)
        return scores

    def decode(self,result):
        return self.config.spam_config.label[result[0][0]]

    def predict(self,X,tokenizer=None,decode=False):
        result = []
        for i in range(len(X)):
            result.append(self.predict_one(X[i], tokenizer,decode))
        return result            

    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model = tf.keras.models.load_model(filename)

    def evaluate(self, X, y):
        y_pred = self.model.predict_classes(X)

        self.print_evaluation("Spam Detector Evaluation",y, y_pred)
        
    def print_evaluation(self, task_name, y_true, y_pred):
        print(task_name)
        print("Precision : ", precision_score(y_true, y_pred, average='macro'))
        print("Recall : ", recall_score(y_true, y_pred, average='macro'))
        print("F1-score : ", f1_score(y_true, y_pred, average='macro'))
        print("Confusion matrix : \n", confusion_matrix(y_true, y_pred))