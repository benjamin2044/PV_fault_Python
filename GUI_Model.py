

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

dataset = pd.read_csv('Solar_categorical.csv')
X = dataset.iloc[:3000, 0:7].values
y = dataset.iloc[:3000, 7].values
encoder= LabelEncoder()
X[:,6] = encoder.fit_transform(X[:, 6])
y = encoder.fit_transform(y)
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

new_model = load_model('fault_model.model')


##### Making Graphical User Interface (GUI) ####
from tkinter import *

root = Tk()
root.title("Fault Detection Model")
root.geometry("320x510+0+0")
root.wm_iconbitmap('icon.ico')

def infoMsg():
        messagebox.askokcancel(title="Help", message="This application was developed by Barun Basnet using tkinter.")

my_menu = Menu(root)

file_menu = Menu(my_menu, tearoff=0)
file_menu.add_command(label="Exit", command=root.destroy)
my_menu.add_cascade(label="File", menu=file_menu)

info_menu = Menu(my_menu, tearoff=0)
info_menu.add_command(label="Info", command=infoMsg)
my_menu.add_cascade(label="About", menu=info_menu)

root.config(menu=my_menu)


heading = Label(root, text="1.8KW Grid-type PV System", font=("arial", 10,"bold"), fg="black").pack()

label1 = Label(root, text="Sensor1 (Amps):", font=("arial", 10,"bold"), fg="green").place(x =10, y=40)
name1 = DoubleVar()
entry_box1 = Entry(root, textvariable=name1).place(x=160, y=40)

label2 = Label(root, text="Sensor2 (Amps):", font=("arial", 10,"bold"), fg="green").place(x =10, y=80)
name2 = DoubleVar()
entry_box2 = Entry(root, textvariable=name2).place(x=160, y=80)

label3 = Label(root, text="Sensor3 (Volts):", font=("arial", 10,"bold"), fg="green").place(x =10, y=120)
name3 = DoubleVar()
entry_box3 = Entry(root, textvariable=name3).place(x=160, y=120)

label4 = Label(root, text="Sensor4 (Volts):", font=("arial", 10,"bold"), fg="green").place(x =10, y=160)
name4 = DoubleVar()
entry_box4 = Entry(root, textvariable=name4).place(x=160, y=160)

label5 = Label(root, text="Irradiance (Klux):", font=("arial", 10,"bold"), fg="green").place(x =10, y=200)
name5 = DoubleVar()
entry_box5 = Entry(root, textvariable=name5).place(x=160, y=200)

label6 = Label(root, text="Temperature (degC):", font=("arial", 10,"bold"), fg="green").place(x =10, y=240)
name6 = DoubleVar()
entry_box6 = Entry(root, textvariable=name6).place(x=160, y=240)

label7 = Label(root, text="Sunny (yes:'0' no:'1'):", font=("arial", 10,"bold"), fg="green").place(x =10, y=280)
name7 = IntVar()
entry_box6 = Entry(root, textvariable=name7).place(x=160, y=280)

def fault_diagnosis():
    ResultBox.delete(0.0, 'end')
    new_mod_test = new_model.predict(sc.transform(np.array([[name1.get(), name2.get(), name3.get(), 
                                                             name4.get(), name5.get(), name6.get(), name7.get()]])))  
    new_mod_test_original = encoder.inverse_transform([np.argmax(new_mod_test)])
    ResultBox.insert(INSERT, new_mod_test_original)

work = Button(root, text="Diagnose", width=20, height=2, bg="lightblue", command=fault_diagnosis).place(x=60, y=340)

ResultBox = Text(root, width=35, height=5)
ResultBox.place(x=10, y=390)


root.mainloop()
