import os
import model.constant as m_const
class Config(object):
    def __init__(self):
        self.module_name = [m_const.Spam, m_const.Emotion, m_const.Spam ]
        self.spam_config = ConfigSpam()
        self.sent_config = ConfigSent()
        self.emot_config = ConfigEmot()

        # Word embeddings
        self.word_dim = 100

        # Word max length
        self.max_len = 100

class BaseConfig(object):
    def __init__(self):
        self.batch_size = 20
        self.epochs = 10

class ConfigSpam(BaseConfig):
    def __init__(self):
        super(ConfigSpam, self).__init__()
        self.label = {
            'ham' : 0,
            'spam' : 1
        }
        self.rnn_cell = 'regu'
        self.hidden_size = 300
        # self.dropout_rate = 0.50
        self.threshold = 0.5

class ConfigSent(BaseConfig):
    def __init__(self):
        super(ConfigSent, self).__init__()
        

class ConfigEmot(BaseConfig):
    def __init__(self):
        super(ConfigEmot, self).__init__()
        