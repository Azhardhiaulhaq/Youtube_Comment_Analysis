import pandas as pd
import nltk
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
from sklearn.metrics import f1_score

class MyEmotionDetector : 
    def __init__(self):
        self.maxlen = 100
        
    def init_model(self,word_embedding):
        self.model = Sequential([
            Embedding(self.vocab_size, 100, weights=[self.embedding_matrix], input_length=self.maxlen , trainable=False),
            Bidirectional(GRU(200, return_sequences=True)),
            Bidirectional(GRU(200,)),
            Dense(1024, activation="relu"),
            Dense(512, activation="relu"),
            Dense(3, activation="softmax")])
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
        # if decode:
        #     return self.decode(scores)
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

        self.print_evaluation("Emotion Detector Evaluation",y, y_pred)
        
    def print_evaluation(self, task_name, y_true, y_pred):
        print(task_name)
        print("Precision : ", precision_score(y_true, y_pred, average='macro'))
        print("Recall : ", recall_score(y_true, y_pred, average='macro'))
        print("F1-score : ", f1_score(y_true, y_pred, average='macro'))
        print("Confusion matrix : \n", confusion_matrix(y_true, y_pred))

    # def load_dataset(self,path):
    #     data = pd.read_csv(path,encoding='latin-1',names=["label","text","none"],header=None)
    #     target_emotion = ['joy','anger','sadness']
    #     data = data.drop(columns=['none'])
    #     data = data.iloc[1:]
    #     # selecting rows based on condition  
    #     dataset = data.loc[data['label'].isin(target_emotion)]
    #     dataset = shuffle(dataset)
    #     return dataset
    
    # def preprocess_text(self,sentence):
    #     return sentence

    # def get_embedding_matrix(self):
    #     embeddings_dictionary = dict()
    #     with open('dataset/glove.6B.100d.txt', encoding="utf8") as glove_file:
    #         for line in glove_file:
    #             records = line.split()
    #             word = records[0]
    #             vector_dimensions = np.asarray(records[1:], dtype='float32')
    #             embeddings_dictionary [word] = vector_dimensions

    #     embedding_matrix = np.zeros((self.vocab_size, 100))
    #     for word, index in self.tokenizer.word_index.items():
    #         embedding_vector = embeddings_dictionary.get(word)
    #         if embedding_vector is not None:
    #             embedding_matrix[index] = embedding_vector
    #     return embedding_matrix

    # def get_model(self):
    #     model = Sequential([
    #         Embedding(self.vocab_size, 100, weights=[self.embedding_matrix], input_length=self.maxlen , trainable=False),
    #         Bidirectional(GRU(200, return_sequences=True)),
    #         Bidirectional(GRU(200,)),
    #         Dense(1024, activation="relu"),
    #         Dense(512, activation="relu"),
    #         Dense(3, activation="softmax")])
    #     return model
    
  

    # def get_tokenizer(self,data_train):
    #     tokenizer = Tokenizer(num_words=10000)
    #     tokenizer.fit_on_texts(data_train)
    #     with open('tokenizer','wb') as handle : 
    #         pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)
    #     return tokenizer

    # def get_prediction(self,list):
    #     data = np.zeros(shape=(list.shape),dtype=int)
    #     data[np.where(list == np.max(list))] = 1
    #     return data.tolist()
    
    # def save_model(self,model,filename):
    #     model.save(filename)

    # def load_model(self,filename):
    #     self.model = tf.keras.models.load_model(filename)

    # def load_tokenizer(self):
    #     with open('tokenizer', 'rb') as handle:
    #         Tokenizer = pickle.load(handle)
    #     return Tokenizer

    # def train(self):
    #     print('-------- Initiate Trainig --------')
    #     dataset = self.load_dataset("dataset/iseardataset.csv")
        
    #     X = []
    #     sentence = list(dataset['text'])
    #     for sen in sentence:
    #         X.append(self.preprocess_text(sen))
    #     y = dataset['label']
    #     encoder = LabelBinarizer()
    #     y = encoder.fit_transform(y)

    #     print(dataset['label'])
    #     print(y)
    #     print('--------------')
    #     # Split train and test data
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    #     # Tokenize sentencs to numbers with max number 10000
    #     self.tokenizer = self.get_tokenizer(X_train)
    #     X_train = self.tokenizer.texts_to_sequences(X_train)
    #     X_test = self.tokenizer.texts_to_sequences(X_test)
    #     # Adding 1 because of reserved 0 index
    #     self.vocab_size = len(self.tokenizer.word_index) + 1
    #     X_train = pad_sequences(X_train, padding='post', maxlen=self.maxlen)
    #     X_test = pad_sequences(X_test, padding='post', maxlen=self.maxlen)
    #     self.embedding_matrix = self.get_embedding_matrix()

    #     self.model = self.get_model()
    #     self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #     self.model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2)
    #     y_pred = self.model.predict(X_test)
    #     result = []
    #     for pred in y_pred:
    #         result.append(self.get_prediction(pred))
    #     f1 = f1_score(y_test,result,average='micro')
    #     print('F1 Score : ')
    #     print(f1)
    #     self.save_model(self.model,"EmotionDetectorModel")


    
    # def predict(self, sentence):
    #     self.load_model("EmotionDetectorModel")
    #     tokenizer = self.load_tokenizer()
    #     print(sentence)
    #     sentence = tokenizer.texts_to_sequences(sentence)
    #     print(sentence)
    #     print(np.array(sentence).shape)
    #     print('------')
    #     sentence = pad_sequences(sentence, padding='post', maxlen=self.maxlen)
    #     print(sentence)
    #     y_pred = self.model.predict(sentence)
    #     y_pred = self.get_prediction(y_pred)
    #     print(y_pred)


emotion = MyEmotionDetector()
# emotion.train()
emotion.predict(['I love that new comments keep adding to it actively. To know that people are listening to it, taking in the information and wisdom, some of us (like myself) keeps coming back to it too. This is such a simple life lesson and yet often so hard to learn. Its the people, not the materials, that truly counts.'])