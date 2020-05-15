import pandas as pd
import cv2
import random
from uuid import uuid1
import os
import PySimpleGUI as sg

left = 'lewa'
center = 'srodek'
right = 'prawa'
left_center = 'lewa-srodek'
right_center = 'prawa-srodek'
collision = 'kolizja'
nocollision = 'brak kolizji'
left_right = 'lewo-prawo'
labels_str = [left, center, right, left_center, right_center, collision, nocollision, left_right]
labels_data = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1], [1,1,1], [0,0,0], [1,0,1]]

layout = [
    [sg.Text('Klasyfikacja zdjęć')],
    [sg.Button(left), sg.Button(center), sg.Button(right)],
    [sg.Button(left_center), sg.Button(right_center)],
    [sg.Button(left_right)],
    [sg.Button(collision)],
    [sg.Button(nocollision)],
]

window = sg.Window('DataLabeler', layout, default_button_element_size=(5, 2), auto_size_buttons=True, grab_anywhere=False)

raw_images = 'raw_images/'
csvfile = 'csvgenerated.csv'
labeled_images = 'labeled_data/'
raw_images_dir = os.listdir(raw_images)

img_num = len(raw_images_dir)


empty_dict = {
    'img_name':[],
    'left':[],
    'center':[],
    'right':[]
}
df = pd.DataFrame(empty_dict)
if not os.path.isfile(csvfile):
    df.to_csv(csvfile)
    df = pd.read_csv(csvfile, index_col=0)
else:
    df = pd.read_csv(csvfile, index_col=0)
print(df)
for i in range(img_num):
    img_path = os.path.join(raw_images, raw_images_dir[i])
    img = cv2.imread(img_path)
    cv2.imshow('plik', img)
    event, values = window.Read()  # read the window
    if event is None:  # if the X button clicked, just exit
        break
    for i in range(len(labels_str)):
        if event == labels_str[i]:
            data = labels_data[i]
            uuid = uuid1()
            os.rename(img_path, os.path.join(labeled_images, str(uuid)))
            n = len(df)
            new_entry = {'img_name': str(uuid), 'left':data[0], 'center':data[1], 'right':data[2]}
            df.loc[n] = new_entry
            df.to_csv(csvfile)