import numpy as np

class FeatureExtractor():
    def __init__(self, config):
        self.config = config
        self.embeddings = None
        self.oov = None

    def init_embedding(self, path):
        self.embeddings = dict()
        with open(path, encoding="utf8") as glove_file:
            for line in glove_file:
                records = line.split()
                word = records[0]
                vector_dimensions = np.asarray(records[1:], dtype='float32')
                self.embeddings[word] = vector_dimensions
        
        self.oov = np.zeros(self.config.word_dim)

    def transform(self, tokens):
        max_len = self.config.max_len
        result = []
        for token in tokens:
            try:
                domain_embedding = self.embeddings[token]
            except KeyError:
                domain_embedding = self.oov

            result.append(domain_embedding)
        
        if max_len != None:
            for _ in range(len(tokens), max_len):
                result.append(self.oov)

        return np.asarray(result)
    
    def get_features(self,X):
        features = []
        for i in range(len(X)):
            features.append(self.transform(X[i]))
        return features
