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
from sklearn.metrics import f1_score, precision_score, recall_score,confusion_matrix
from emoji import UNICODE_EMOJI

class MyEmotionDetector : 
    def __init__(self):
        self.maxlen = 100

    def save_model(self, filename):
        self.model.save(filename)

    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        result = []
        for pred in y_pred:
            result.append(self.get_prediction(pred))

        self.print_evaluation("Emotion Detector Evaluation",y, result)
        
    def print_evaluation(self, task_name, y_true, y_pred):
        print(task_name)
        print("Precision : ", precision_score(y_true, y_pred, average='macro'))
        print("Recall : ", recall_score(y_true, y_pred, average='macro'))
        print("F1-score : ", f1_score(y_true, y_pred, average='macro'))



    def load_dataset(self,path):
        data = pd.read_csv(path,encoding='latin-1',names=['comment','sentiment','emotion','spam'],header=None)
        data = data.drop(columns=['sentiment','spam'])
        data = data.iloc[1:]
        return data
    
    def is_emoji(self,target_emoji,list_emoji):
        count = 0
        for emoji in list_emoji:
            count += target_emoji.count(emoji)
            if count > 1:
                return False
        return bool(count)

    def emoji_classification(self,token):
        joy_emoji = "👏😀😃😄😁😆😅🤣😂❤️💚🤭😘🤡💙👍😍💋 💝💖♥️❤️"
        anger_emoji = "😠😤🤬💀☠️🙅🏼‍"
        sad_emoji = "😥😭😢😰😓💔🤦‍😓"
        neutral_emoji = "🤔😐😶"
        if(self.is_emoji(token,joy_emoji)):
            return "joy"
        elif(self.is_emoji(token,anger_emoji)):
            return "anger"
        elif(self.is_emoji(token,sad_emoji)):
            return "sad"
        elif(self.is_emoji(token,neutral_emoji)) :
            return "neutral"       
        else : 
            return token

    def process_emoji(self,tokens):
        result = []
        for token in tokens :
            if (token in UNICODE_EMOJI):
                token = self.emoji_classification(token)
                result.append(token)
            else :
                result.append(token.lower())
            
        return result

    def preprocess_text(self,sentence):
        sentence = sentence[0]
        words = sentence.split()
        words = self.process_emoji(words)
        sentence = ' '.join(map(str,words))
        return [sentence]

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
            Bidirectional(GRU(200, return_sequences=True)),
            Bidirectional(GRU(200,)),
            Dense(1024, activation="relu"),
            Dense(512, activation="relu"),
            Dense(4, activation="sigmoid")])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def get_tokenizer(self,data_train):
        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(data_train)
        with open('modules/model/tokenizer_emot.pickle','wb') as handle : 
            pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)
        return tokenizer

    def get_prediction(self,list):
        data = np.zeros(shape=(list.shape),dtype=int)
        data[np.where(list == np.max(list))] = 1
        return data.tolist()
    
    def load_model(self,filename):
        self.model = tf.keras.models.load_model(filename)
        self.tokenizer = self.load_tokenizer()
        self.encoder = self.load_encoder()

    def load_tokenizer(self):
        with open('modules/model/tokenizer_emot.pickle', 'rb') as handle:
            Tokenizer = pickle.load(handle)
        return Tokenizer
    
    def load_encoder(self):
        with open('modules/model/encoder_emot.pickle', 'rb') as handle:
            Encoder = pickle.load(handle)
        return Encoder

    def preprocessing(self,dataset):
        X = []
        sentence = list(dataset['comment'])
        for sen in sentence:
            X.append(self.preprocess_text(sen))
        y = dataset['emotion']
        encoder = LabelBinarizer()
        y = encoder.fit_transform(y)
        with open('modules/model/encoder_emot.pickle','wb') as handle : 
            pickle.dump(encoder,handle,protocol=pickle.HIGHEST_PROTOCOL)

        return X,y

    def feature_extraction(self,X_train,X_test):
        X_train = self.tokenizer.texts_to_sequences(X_train)
        X_test = self.tokenizer.texts_to_sequences(X_test)
        # # Adding 1 because of reserved 0 index
        self.vocab_size = len(self.tokenizer.word_index) + 1
        X_train = pad_sequences(X_train, padding='post', maxlen=self.maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=self.maxlen)
        return X_train,X_test

    def train(self):
        dataset = self.load_dataset('dataset/youtube_comment_dataset.csv')
        
        print(dataset.emotion.value_counts())
        X,y = self.preprocessing(dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        
        # # Tokenize sentencs to numbers with max number 10000
        self.tokenizer = self.get_tokenizer(X_train)
        X_train,X_test = self.feature_extraction(X_train,X_test)
        self.embedding_matrix = self.get_embedding_matrix()

        self.model = self.get_model()
        print(self.model.summary())
        self.model.fit(X_train, y_train, batch_size=64, epochs=15, validation_split=0.1)
        self.save_model("modules/model/EmotionDetectorModel")
        self.evaluate(X_test,y_test)



    
    def predict(self, sentence):
        sentence = self.preprocess_text(sentence)
        sentence = self.tokenizer.texts_to_sequences(sentence)
        sentence = pad_sequences(sentence, padding='post', maxlen=self.maxlen)
        y_pred = self.model.predict(sentence)
        y_pred = self.get_prediction(y_pred)
        emotion = self.encoder.inverse_transform(np.array(y_pred))
        return emotion[0]


# # emotion = MyEmotionDetector()
# # # emotion.train()
# # emotion.predict(['I love that new comments keep adding to it actively. To know that people are listening to it, taking in the information and wisdom, some of us (like myself) keeps coming back to it too. This is such a simple life lesson and yet often so hard to learn. Its the people, not the materials, that truly counts.'])

# # y = ['Neutral','Joy','Sad','Angry','Neutral','Joy']
# # print(y)
# # encoder = LabelBinarizer()
# # y = encoder.fit_transform(y)
# # print(y)
# # print(encoder.inverse_transform(y))

# emotion = MyEmotionDetector()
# # # emotion.train()
# emotion.load_model("model/EmotionDetectorModel.model")
# emotion.predict(['Im crying seeing this'])

# def is_emoji(s):
#     emojis = "😘💙" # add more emojis here
#     count = 0
#     for emoji in emojis:
#         count += s.count(emoji)
#         if count > 1:
#             return False
#     return bool(count)

# text = ["Recommendations are faster than notification 😂"]
# emotion = MyEmotionDetector()
# words = emotion.preprocess_text(text)
# print(words)