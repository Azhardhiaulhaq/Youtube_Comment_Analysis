from model.ses import SentimentEmotionSpamDetect
from config import Config
import model.constant as consts
import utils

if __name__ == "__main__":
    config = Config()
    print(config.spam_config.batch_size)
    # model = SentimentEmotionSpamDetect(config)

    # Load Dataset
    path = 'dataset/Youtube01-Psy.csv'
    glove_path = 'word_embeddings/glove.6B.100d.txt'
    df = utils.load_dataset(path)
    X_train, y_train, X_test, y_test, word_embed_train = utils.prep_data(df,config,glove_path)

    model = SentimentEmotionSpamDetect(config)
    model.init_model(word_embed_train)
    # Train Spam
    model.train(X_train,y_train, m=consts.Spam)
    model.save_model('saved_model/coba2')

    # Train Sentiment

    # Train Emotion