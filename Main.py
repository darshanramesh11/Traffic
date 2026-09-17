import tkinter as tk
from tkinter import Message ,Text
#import cv2,os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font
import RandomForest as rand
import kNNClassifier as kn


from matplotlib import pyplot as plt



window = tk.Tk()
window.title("Traffic Prediction")

 
window.geometry('1280x720')
window.configure(background='red')

#window.attributes('-fullscreen', True)

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)


message1 = tk.Label(window, text="Traffic Prediction" ,bg="blue"  ,fg="white"  ,width=50  ,height=2,font=('times', 30, 'italic bold underline')) 
message1.place(x=100, y=20)



def knn():
	print("K Nearest Neighbor")
	kn.knn()
	
def rf():
	print("Random Forest")
	rand.rf()



trainImg = tk.Button(window, text="PREDICT Using KNN Classifier", command=knn  ,fg="red"  ,bg="yellow"  ,width=30  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
trainImg.place(x=400, y=200)

detect = tk.Button(window, text="PREDICT Using Random Forest", command=rf  ,fg="red"  ,bg="yellow"  ,width=30  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
detect.place(x=700, y=400)

quitWindow = tk.Button(window, text="QUIT", command=window.destroy  ,fg="black"  ,bg="yellow"  ,width=20  ,height=3, activebackground = "Red" ,font=('times', 15, ' bold '))
quitWindow.place(x=900, y=600)

 
window.mainloop()