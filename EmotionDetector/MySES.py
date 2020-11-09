from EmotionDetector.MyEmotionDetector import MyEmotionDetector
from EmotionDetector.MySpamDetector import MySpamDetector

class MySES(object):
    def __init__(self):
        self.modules = [None, None, MySpamDetector()]
        self.name = ['sent', 'emot', 'spam']

    def train(self):
        for module in self.modules:
            if module is not None:
                module.train()
    
    def load_model(self, filename_list):
        for filename, module  in zip(filename_list, self.modules):
            if module is not None:
                module.load_model(filename)

    def predict(self, sentence):
        result = []
        for name, module in zip(self.name,self.modules):
            if module is not None:
                result.append({name : module.predict(sentence)})
            else:
                result.append({name : ["TODO"]})
        return result