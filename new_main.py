from EmotionDetector.MySES import MySES

if __name__ == "__main__":
    model = MySES()
    model.load_model([None, None,'saved_model/SpamDetectorModel'])
    test = model.predict("These TED TALKS are worth watching in this time of pandemic")
    print(test)