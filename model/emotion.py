import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, Dense, GRU
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

class EmotionModule(object) :
    def __init__(self):
        self.maxlen = 100
        
    def init_model(self,word_embedding):
        self.model = Sequential([
            Embedding(word_embedding.vocab_size, word_embedding.dim, weights=[word_embedding.embeddings_matrix], input_length=self.maxlen, trainable=False),
            Bidirectional(GRU(200, return_sequences=True)),
            Bidirectional(GRU(200,)),
            Dense(1024, activation="relu"),
            Dense(512, activation="relu"),
            Dense(4, activation="softmax")])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self,X_train,y_train):
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
        scores = self.model.predict(X)
        print(scores)
        if decode:
            return self.decode(scores)
        return scores

    def get_prediction_class(self,list):
        data = np.zeros(shape=(list.shape),dtype=int)
        data[np.where(list == np.max(list))] = 1
        return data.tolist()

    def decode(self,result):
        result = self.get_prediction_class(result)
        return result

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
        y_pred = self.model.predict(X)
        y_pred = self.get_prediction_class(y_pred)
        self.print_evaluation("Emotion Detector Evaluation",y, y_pred)
        
    def print_evaluation(self, task_name, y_true, y_pred):
        print(task_name)
        print("Precision : ", precision_score(y_true, y_pred, average='macro'))
        print("Recall : ", recall_score(y_true, y_pred, average='macro'))
        print("F1-score : ", f1_score(y_true, y_pred, average='macro'))
        print("Confusion matrix : \n", confusion_matrix(y_true, y_pred))