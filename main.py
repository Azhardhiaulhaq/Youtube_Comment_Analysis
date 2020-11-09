from model.ses import SentimentEmotionSpamDetect
from config import Config
import model.constant as consts
import utils
from tensorflow.keras.backend import clear_session
if __name__ == "__main__":
    clear_session()
    config = Config()
    
    ses = SentimentEmotionSpamDetect(config)
    # ses.load_model('saved_model/coba2spam',consts.Spam)

    tokenizer = utils.load_tokenizer('tokenizer')

    text = ["These  election is the worst ever"]
    # text = utils.preprocess_text(text, tokenizer, config)
    print(text)
    preprocess_text = utils.get_preprocessor_func(tokenizer, config)
    # print(text.shape)
    result = ses.predict_one(text,preprocess_text, decode=True)
    print(result)
