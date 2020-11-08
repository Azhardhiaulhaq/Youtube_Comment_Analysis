import numpy as np

class FeatureExtractor():
    def __init__(self, word_embeddings_model, config):
        self.embeddings = self.init_embedding(word_embeddings_model)
        self.oov = np.zeros(config.word_dim)
    
    def init_embedding(self, word_embeddings_model):
        embeddings_dictionary = dict()
        with open(word_embeddings_model, encoding="utf8") as glove_file:
            for line in glove_file:
                records = line.split()
                word = records[0]
                vector_dimensions = np.asarray(records[1:], dtype='float32')
                embeddings_dictionary[word] = vector_dimensions
        return embeddings_dictionary

    def transform(self, tokens, max_len=None):
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
    
    def get_features(self,X, max_len=None):
        features = []
        for i in range(len(X)):
            features.append(self.transform(X[i], max_len))

        return features
