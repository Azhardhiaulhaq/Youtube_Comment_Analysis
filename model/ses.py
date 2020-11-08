from model.spam import SpamModules
import model.constant as consts

class SentimentEmotionSpamDetect(object):
    def __init__(self,config):
        self.config = config
        # Sentiment, Emotion, Spam
        self.modules = [self.instance_model(module) for module in self.config.module_name]
    
    def instance_model(self, m):
        if m == consts.Sentiment :
            return None
        elif m == consts.Spam :
            return SpamModules(self.config)
        elif m == consts.Emotion :
            return None
        else:
            raise ValueError(consts.UnknownModuleError)

    def init_model(self, word_embedding):
        for module in self.modules:
            if module is not None:
                module.init_model(word_embedding)

    def get_index_module(self, m):
        if m in self.config.module_name:
            return self.config.module_name.index(m)
        raise ValueError(consts.UnknownModuleError)

    def train(self, X_train, y_train, m=consts.Spam):
        idx = self.get_index_module(m)
        self.modules[idx].train(X_train, y_train)
    
    def load_model(self, path, m=consts.Spam):
        idx = self.get_index_module(m)
        self.modules[idx].load_model(path)
    
    def predict_one(self, sentence,feature_extractor=None,decode=False):
        result = []
        for name, module in zip(self.config.module_name,self.modules):
            if module is not None:
                result.append({name : module.predict_one(sentence,feature_extractor, decode)})
            else:
                result.append({name : ["TODO"]})
    
    def predict(self, X,feature_extractor=None,decode=False):
        result = []
        for sentence in X:
            result.append(self.predict_one(sentence,feature_extractor,decode))
    
    def save_model(self, base_filename):
        for name in self.config.module_name:
            self.save_module_one(base_filename, name)
    
    def save_module_one(self, base_filename, m):
        idx = self.get_index_module(m)
        if self.modules[idx] is not None:
            self.modules[idx].save_model(base_filename+m)