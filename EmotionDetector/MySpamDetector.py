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

class MySpamDetector(object):
    def __init__(self):
        self.maxlen = 100
    
    def load_dataset(self,path):
        data = pd.read_csv(path, encoding='latin-1')
        