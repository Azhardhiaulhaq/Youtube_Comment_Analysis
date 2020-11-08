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
from sklearn import metrics

class SpamModules(object):
    def __init__(self,config):
        self.config = config
        self.model = None
    
    def init_model(self):
        self.model = Sequential([
            layers.Embedding(),
            layers.Bidirectional(),
            layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
    def train(self,X_train,y_train,X_val=None,y_val=None):
        history = self.model.fit(X_train, y_train, X_val=X_val,y_val=y_val)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig('spam_loss')
    
    def predict_one(self,sentence):
        pass

    def save_model(self, filename):
        pass

    def load_model(self, filename):
        pass
    
    def evaluate(self, X, y):
        pass
