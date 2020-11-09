from tkinter import *
from tkinter.ttk import *

class GUI(Frame):

    def __init__(self,master = None):
        Frame.__init__(self,master)
        self.master = master
        self.init_GUI()

    def init_GUI(self):
        self.master.title("Youtube Sentiment Analysis")
        self.pack(fill=BOTH,expand=1)
        # creating a button instance


        style = Style() 
        style.configure('W.TButton', font= ('Arial', 10, 'underline'),
            foreground='Green')



        textLabel = Label(self,text = "Input Text")
        textLabel.grid(row = 0, column = 0,ipadx=10,ipady=10)

        E1 = Entry(self)
        E1.grid(row=1,column = 0,ipadx=20,ipady=20,padx=100,rowspan=10)        

        predictButton = Button(self, style='W.TButton',text="Predict", command = self.client_exit)
        predictButton.grid(row=0,column = 1)
        # E2 = Entry(self)
        # E2.grid(row=1)
    def client_exit(self):
        exit()


root = Tk()

#size of the window
root.geometry("800x600")
app = GUI(root)
root.mainloop() 