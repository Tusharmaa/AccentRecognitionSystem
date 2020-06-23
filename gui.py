from PIL import ImageTk,Image
import tkinter as tk
from tkinter import *
from tkinter import filedialog
image_size = 128
from PIL import Image, ImageTk
import os
import pandas as pd
from collections import Counter
import sys
sys.path.append('../speech-accent-recognition/src>')
import getsplit
from keras import utils
import accuracy
import multiprocessing
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import test
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard
from keras.models import load_model
import speech_recognition as sr
print(librosa.__version__)
import pyaudio
import wave






#GUI
root = Tk()  # Main window
f = Frame(root)
frame1 = Frame(root)
frame2 = Frame(root)
frame3 = Frame(root)
root.title("Accent Recognition")
root.geometry("1080x720")

canvas = Canvas(width=1080, height=250)
canvas.pack()
filename=('accent.png')
load = Image.open(filename)
load = load.resize((1800, 250), Image.ANTIALIAS)
render = ImageTk.PhotoImage(load)
img = Label(image=render)
img.image = render
#photo = PhotoImage(file='landscape.png')
load = Image.open(filename)
img.place(x=1, y=1)
#canvas.create_image(-80, -80, image=img, anchor=NW)

root.configure(background='#a6a6a6')
scrollbar = Scrollbar(root)
scrollbar.pack(side=RIGHT, fill=Y)

firstname = StringVar()  # Declaration of all variables
lastname = StringVar()
id = StringVar()
dept = StringVar()
designation = StringVar()
remove_firstname = StringVar()
remove_lastname = StringVar()
searchfirstname = StringVar()
searchlastname = StringVar()
sheet_data = []
row_data = []



def add_entries():  # to append all data and add entries on click the button
    a = " "
    f = firstname.get()
    f1 = f.lower()
    l = lastname.get()
    l1 = l.lower()
    d = dept.get()
    d1 = d.lower()
    de = designation.get()
    de1 = de.lower()
    list1 = list(a)
    list1.append(f1)
    list1.append(l1)
    list1.append(d1)
    list1.append(de1)


def click( ):
    textbox.delete('1.0',"end-1c")
    #filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype =(("all files","*.*"),("jpeg files","*.jpg")) )
    filename = filedialog.askopenfilename(message ='r', filetypes =[('wave Files', '*.wav')])


    textbox.delete('1.0', "end-1c")
    textbox.insert("end-1c", filename)
    fname=os.path.basename(filename)
    test.audioname(fname)
    output=test.main()
    if(output[0]==0):
        textbox1.insert("end-1c", "Mandarian")
    elif(output[0]==1):
        textbox1.insert("end-1c", "English")
    else:
        textbox1.insert("end-1c", "Arabic")


def speak():

    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3
    WAVE_OUTPUT_FILENAME = "audio_ORG/audio.wav"

            # start Recording
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,channels=CHANNELS, rate=RATE, input=True,frames_per_buffer=CHUNK)

    print("* recording")
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("* done")

            # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    filename = 'audio.wav'
    textbox.delete('1.0', "end-1c")
    textbox.insert("end-1c", filename)
    fname=os.path.basename(filename)
    test.audioname(fname)
    output=test.main()
    if(output[0]==0):
        textbox1.insert("end-1c", "Mandarian")
    elif(output[0]==1):
        textbox1.insert("end-1c", "English")
    else:
        textbox1.insert("end-1c", "Arabic")


def clear_all():  # for clearing the entry widgets
    frame1.pack_forget()
    frame2.pack_forget()
    frame3.pack_forget()


label1 = Label(root, text="Accent Recognition ")
label1.config(font=('Italic', 18, 'bold'), justify=CENTER, background='#00bfff', fg="Black", anchor="center")
label1.pack(fill=X)


frame2.pack_forget()
frame3.pack_forget()

textbox = tk.Text(frame1,width="30",height=1)
textbox.grid(row=1, column=1,padx=10,pady=10)

button4 = Button(frame1, text="Browse", command=click)
button4.grid(row=1, column=2,padx=10,pady=10)

button5 = Button(frame1, text="Speak", command=speak)
button5.grid(row=2, column=1,padx=10,pady=10)

textbox1 = tk.Text(frame1,width="30",height=1,padx=10 )
textbox1.grid(row=3, column=1,padx=10,pady=10)

frame1.configure(background="black")
frame1.pack(pady=10)

root.mainloop()
