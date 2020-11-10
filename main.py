from modules.MySES import MySES
from gui import GUI
from tkinter import *
from tkinter.ttk import *

if __name__ == "__main__":
    # model = MySES()
    # model.load_model(["modules/model/SentimentDetectorModel", "modules/model/EmotionDetectorModel","modules/model/SpamDetectorModel"])
    # test = model.predict(["These TED TALKS are worth watching in this time of pandemic"])
    # print(test)
    root = Tk()
    #size of the window
    root.geometry("800x600")
    app = GUI(root)
    root.mainloop() 
