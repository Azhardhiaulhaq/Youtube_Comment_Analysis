import numpy as np

class FeatureExtractor():
    def __init__(self, config):
        self.config = config
        self.tokenizer = None

    def init_feat(self, t):
        self.tokenizer =  None

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
