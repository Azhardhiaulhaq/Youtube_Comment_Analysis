from model.ses import SentimentEmotionSpamDetect
from config import Config
import model.constant as consts

if __name__ == "__main__":
    config = Config()
    model = SentimentEmotionSpamDetect(config)

    # Load Dataset

    # Train Spam
    model.fit()

    # Train Sentiment

    # Train Emotion