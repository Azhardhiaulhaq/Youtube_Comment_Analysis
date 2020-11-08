import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from word_embeddings.word_embeddings import WordEmbedder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle

def load_dataset(path):
    df = pd.read_csv(path,encoding='latin-1')
    return df

# TODO
def load_tokenizer(path):
    with open(path, 'rb') as handle:
        Tokenizer = pickle.load(handle)
    return Tokenizer

def get_preprocessor_func(tokenizer, config):
    return lambda t: preprocess_text(t, tokenizer, config)

def preprocess_text(sentence, tokenizer=None, config=None):
    if tokenizer is None:
        return sentence
    else:
        sentence = tokenizer.texts_to_sequences(sentence)
        sentence = pad_sequences(sentence, padding='post', maxlen=config.max_len)
        return sentence

def get_tokenizer(data_train):
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(data_train)
    with open('tokenizer','wb') as handle : 
        pickle.dump(tokenizer,handle,protocol=pickle.HIGHEST_PROTOCOL)
    return tokenizer

def get_spam_label(df, label ):
    return df[label]
# return X_train, y_train, X_test, y_test (y bisa dibagi jadi y_spam_train, y_sent_train , dst)
def prep_data(df, config,path_word_embeddings, tokenizer_path=None):
    
    X = []
    sentence = list(df['CONTENT'])
    for sen in sentence:
        X.append(preprocess_text(sen))
    y_spam = get_spam_label(df, ['CLASS'])
    # encoder = LabelBinarizer()
    # y_spam = encoder.fit_transform(y_spam)

    # Split train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y_spam, test_size=0.20, random_state=42)

    # Tokenize sentencs to numbers with max number 10000
    tokenizer = get_tokenizer(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    X_train = pad_sequences(X_train, padding='post', maxlen=config.max_len)
    X_test = pad_sequences(X_test, padding='post', maxlen=config.max_len)

    wordembedding = WordEmbedder(config)
    wordembedding.init_embedding(path_word_embeddings, tokenizer)

    return X_train, y_train, X_test, y_test, wordembedding

