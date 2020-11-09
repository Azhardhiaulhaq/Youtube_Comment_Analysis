import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, Dense
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

class SentimentModule(object) :
    def __init__(self) :
        self.maxlen = 100