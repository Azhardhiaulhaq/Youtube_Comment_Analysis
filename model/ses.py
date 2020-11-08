from model.spam import SpamModules
import model.constant as consts

class SentimentEmotionSpamDetect(object):
    def __init__(self,config):
        self.config = config
        # Sentiment, Emotion, Spam
        self.modules = [self.instance_model(module) for module in self.config.modules]
    
    def instance_model(self, m):
        if m == consts.Sentiment :
            return None
        elif m == consts.Spam :
            return SpamModules()
        elif m == consts.Emotion :
            return None
        else:
            raise ValueError(consts.UnknownModuleError)


    def init_model(self):
        for module in self.modules:
            if module is not None:
                module.init_model()

    def get_index_module(self, m):
        if m in self.config.modules:
            return self.config.modules.index(m)
        raise ValueError(consts.UnknownModuleError)

    def train(self, X_train, y_train, m=consts.Spam, X_val=None, y_val=None ):
        idx = self.get_index_module(m)
        self.modules[idx].train(X_train, y_train, X_val=X_val, y_val=y_val)
    
    def load_model(self, path, m=consts.Spam):
        idx = self.get_index_module(m)
        self.modules[idx].load_model(path)
    
    def predict_one(self, sentence):
        result = []
        for name, module in zip(self.config.modules,self.modules):
            if module is not None:
                result.append({name : module.predict_one(sentence)})
            else:
                result.append({name : ["TODO"]})
    
    def predict(self, X):
        result = []
        for sentence in X:
            result.append(self.predict_one(sentence))
