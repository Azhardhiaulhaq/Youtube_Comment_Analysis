from EmotionDetector.MySES import MySES

if __name__ == "__main__":
    model = MySES()
    model.load_model([None, "modules/model/EmotionDetectorModel.model",None])
    test = model.predict("These TED TALKS are worth watching in this time of pandemic")
    print(test)