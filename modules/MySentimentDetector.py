import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, Dense, GRU
from sklearn.metrics import f1_score, precision_score, recall_score, multilabel_confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

class MySentimentDetector(object) :
    def __init__(self) :
        self.maxlen = 100

    def load_dataset(self, path) :
        data = pd.read_csv(path, encoding='latin-1', names=['comment', 'sentiment', 'emotion', 'spam'], header=None)
        data = data.drop(columns=['emotion', 'spam'])
        data = data.iloc[1:]
        return data

    def preprocess_text(self, sentence) :
        return sentence

    def get_embedding_matrix(self) :
        embeddings_dictionary = dict()
        with open('dataset/glove.6B.100d.txt', encoding="utf8") as glove_file :
            for line in glove_file :
                records = line.split()
                word = records[0]
                vector_dimensions = np.asarray(records[1:], dtype='float32')
                embeddings_dictionary[word] = vector_dimensions

        embedding_matrix = np.zeros((self.vocab_size, 100))
        for word, index in self.tokenizer.word_index.items() :
            embedding_vector = embeddings_dictionary.get(word)
            if embedding_vector is not None :
                embedding_matrix[index] = embedding_vector
        return embedding_matrix

    def get_model(self) :
        model = Sequential([
            Embedding(self.vocab_size, 100, weights=[self.embedding_matrix], input_length=self.maxlen, trainable=False),
            Bidirectional(GRU(200, return_sequences=True)),
            Bidirectional(GRU(200)),
            Dense(1024, activation="relu"),
            Dense(512, activation="relu"),
            Dense(3, activation="softmax")
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def get_tokenizer(self, data_train) :
        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(data_train)
        with open('modules/model/tokenizer_sent.pickle', 'wb') as handle :
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return tokenizer

    def load_tokenizer(self) :
        with open('modules/model/tokenizer_sent.pickle', 'rb') as handle :
            Tokenizer = pickle.load(handle)
        return Tokenizer

    def load_encoder(self) :
        with open('modules/model/tokenizer_sent.pickle', 'rb') as handle :
            Encoder = pickle.load(handle)
        return Encoder

    def get_prediction(self, list) :
        data = np.zeros(shape=(list.shape), dtype=int)
        data[np.where(list == np.max(list))] = 1
        return data.tolist()

    def save_model(self, filename) :
        self.model.save(filename)

    def load_model(self, filename) :
        self.model = tf.keras.models.load_model(filename)
        self.tokenizer = self.load_tokenizer()
        self.encoder = self.load_encoder()

    def preprocessing(self, dataset) :
        X = []
        sentences = list(dataset['comment'])
        for sentence in sentences :
            X.append(self.preprocess_text(sentence))
        y = dataset['sentiment']
        encoder = LabelBinarizer()
        y = encoder.fit_transform(y)
        with open('modules/model/encoder_sent.pickle', 'wb') as handle :
            pickle.dump(encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return X, y

    def feature_extraction(self, X_train, X_test) :
        X_train = self.tokenizer.texts_to_sequences(X_train)
        X_test = self.tokenizer.texts_to_sequences(X_test)

        # Adding 1 because of reserved 0 index
        self.vocab_size = len(self.tokenizer.word_index) + 1
        X_train = pad_sequences(X_train, padding='post', maxlen=self.maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=self.maxlen)

        return X_train, X_test

    def train(self) :
        dataset = self.load_dataset('dataset/youtube_comment_dataset.csv')

        print(dataset.sentiment.value_counts())
        X, y = self.preprocessing(dataset)

        # Split train and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Tokenize sentences to numbers with max number 10000
        self.tokenizer = self.get_tokenizer(X_train)
        X_train, X_test = self.feature_extraction(X_train, X_test)
        self.embedding_matrix = self.get_embedding_matrix()

        # Initialize and train model
        self.model = self.get_model()
        self.model.fit(X_train, y_train, batch_size=32, epochs=15, validation_split=0.2)
        self.save_model("modules/model/SentimentDetectorModel")
        self.evaluate(X_test, y_test)

    def predict(self, sentence) :
        sentence = self.tokenizer.texts_to_sequences(sentence)
        sentence = pad_sequences(sentence, padding='post', maxlen=self.maxlen)
        y_pred = self.model.predict(sentence)
        y_pred = self.get_prediction(y_pred)
        sentiment = self.encoder.inverse_transform(np.array(y_pred))
        print(sentiment)

    def evaluate(self, X, y) :
        y_pred = self.model.predict(X)
        result = []
        for pred in y_pred :
            result.append(self.get_prediction(pred))
        self.print_evaluation("Sentiment Detector Evaluation", y, result)

    def print_evaluation(self, task_name, y_true, y_pred) :
        print(task_name)
        print("Precision : ", precision_score(y_true, y_pred, average='macro'))
        print("Recall : ", recall_score(y_true, y_pred, average='macro'))
        print("F1-score : ", f1_score(y_true, y_pred, average='macro'))
        print("Confusion matrix : \n", multilabel_confusion_matrix(y_true, y_pred))

# sentiment = MySentimentDetector()
# sentiment.train()
# sentiment.predict(['The quality of this video is really great! i love it'])