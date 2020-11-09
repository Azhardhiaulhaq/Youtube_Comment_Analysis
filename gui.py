from tkinter import *
from tkinter.ttk import *
from PIL import Image, ImageTk

class GUI(Frame):

    def __init__(self,master = None):
        Frame.__init__(self,master)
        self.master = master
        self.init_GUI()

    def init_GUI(self):
        self.master.title("Youtube Sentiment Analysis")
        self.pack()

        text_label = Label(self.master,text = "Insert Text",font = ('helvetica',16,'bold'))
        text_label.place(relx=0.05,rely=0.1)

        self.text_entry = Entry(self.master, font=('helvetica',12))
        self.text_entry.place(relx=0.05,rely=0.15,relheight = 0.7, relwidth = 0.4)  

        button_style = Style() 
        button_style.configure('W.TButton', font= ('Helvetica', 12),
            foreground='black')
        
        self.predict_button = Button(self.master, style='W.TButton',text="Predict", command = self.predict)
        self.predict_button.place(relx = 0.05,rely=0.9,height=30,relwidth = 0.4)

        line_style = Style()
        line_style.configure("Line.TSeparator", background="#000000")
        line = Separator(self.master, orient=VERTICAL, style="Line.TSeparator")
        line.place(relx=0.5, height = self.master.winfo_screenheight(), width = 4)

        result_label = Label(self.master, text = "Result : ",font = ('helvetica',16,'bold'))
        result_label.place(relx=0.55,rely=0.1)

        self.load_sentimen_icon()
        self.load_emotion_icon()
        self.load_spam_icon()


    def load_label(self,path):
        icon = Image.open(path)
        icon = icon.resize((100,100))
        icon = ImageTk.PhotoImage(icon)
        label = Label(self.master,image=icon)
        label.image = icon
        return label

    def load_sentimen_icon(self):
        self.sentimen_label = Label(self.master, text = "Sentiment",font = ('helvetica',14,'bold'))
        self.positive_label = self.load_label("icon/plus_icon.png")
        self.neutral_label = self.load_label("icon/neutral_icon.png")
        self.negative_label = self.load_label("icon/negative_icon.png")

    def load_emotion_icon(self):
        self.emotion_label = Label(self.master, text = "Emotion",font = ('helvetica',14,'bold'))
        self.angry = self.load_label("icon/angry.png")
        self.happy = self.load_label("icon/happy.png")
        self.neutral = self.load_label("icon/neutral.png")
        self.sad = self.load_label("icon/sad.png")

    def load_spam_icon(self):
        self.spam_label = Label(self.master, text = "Spam",font = ('helvetica',14,'bold'))
        self.spam = self.load_label("icon/spam.png")
        self.ham = self.load_label("icon/email.png")

    def show_sentimen(self,sentimen):
        self.sentimen_label.place(relx=0.57,rely=0.44)
        if(sentimen == "positive"):
            self.positive_label.place(relx=0.56,rely=0.25)
            self.neutral_label.place_forget()
            self.negative_label.place_forget()
        elif(sentimen == "neutral"):
            self.neutral_label.place(relx=0.57,rely=0.27)
            self.positive_label.place_forget()
            self.negative_label.place_forget()
        else :
            self.negative_label.place(relx=0.57,rely=0.25)
            self.neutral_label.place_forget()
            self.positive_label.place_forget()
    
    def show_spam(self,spam):
        self.spam_label.place(relx=0.83,rely=0.44)
        if(spam == "spam"):
            self.spam.place(relx=0.8,rely=0.25)
            self.ham.place_forget()
        else : 
            self.ham.place(relx=0.8,rely=0.25)
            self.spam.place_forget()

    def show_emotion(self,emotion):
        self.emotion_label.place(relx=0.69,rely=0.75)
        if(emotion == "Joy"):
            self.happy.place(relx=0.68,rely=0.55)
            self.neutral.place_forget()
            self.sad.place_forget()
            self.angry.place_forget()
        elif(emotion=="Neutral"):
            self.neutral.place(relx=0.68,rely=0.55)
            self.happy.place_forget()
            self.sad.place_forget()
            self.angry.place_forget()
        elif(emotion=="Sad"):
            self.sad.place(relx=0.68,rely=0.55)
            self.happy.place_forget()
            self.neutral.place_forget()
            self.angry.place_forget()
        else : 
            self.angry.place(relx=0.68,rely=0.55)
            self.happy.place_forget()
            self.sad.place_forget()
            self.neutral.place_forget()

    def client_exit(self):
        exit()

    def predict(self):
        sentimen = "negative"
        spam = "ham"
        emotion = "Joy"
        self.show_emotion(emotion)
        self.show_sentimen(sentimen)
        self.show_spam(spam)

root = Tk()

#size of the window
root.geometry("800x600")
app = GUI(root)
root.mainloop() 