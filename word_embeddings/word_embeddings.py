import numpy as np

class WordEmbedder(object):

    def __init__(self, config):
        self.config = config
        self.embeddings_matrix = None
        self.vocab_size = None
        self.dim = config.word_dim

    def init_embedding(self, word_embeddings_model, t):
        embeddings_index = dict()
        with open(word_embeddings_model, encoding="utf8") as glove_file:
            for line in glove_file:
                records = line.split()
                word = records[0]
                vector_dimensions = np.asarray(records[1:], dtype='float32')
                embeddings_index[word] = vector_dimensions
        
        self.vocab_size = len(t.word_index) + 1

        self.embeddings_matrix = np.zeros((self.vocab_size, self.config.word_dim))
        for word, i in t.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                self.embeddings_matrix[i] = embedding_vector
