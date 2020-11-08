import os
import model.constant as m_const
class Config(object):
    def __init__(self):
        self.module_name = [m_const.Spam, m_const.Emotion, m_const.Spam ]
        self.spam_config = ConfigSpam()


class ConfigSpam(object):
    def __init__(self):
        # Model
        self.rnn_cell = 'regu'
        self.hidden_size = 300
        self.dropout_rate = 0.50