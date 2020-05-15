import pandas as pd
import os
import torch
import numpy as np
import cv2
import Utils

possible_labels = [
    [0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[0,1,1],[1,0,1],[1,1,1]
]

#imgname,left,center,right,nocollision
csv_file = 'collision_labels_clean.csv'
root_dir = 'collision/'
root_data = 'data/'
#os.path.join(root_dir, csv_file)
dataset = pd.read_csv(os.path.join(root_dir, csv_file))


def printStats():
    label_values = dataset.iloc[:, 2:6].values
    # Statystyka - Aktualny balans danych
    label_len = np.sum(label_values, axis=0)
    print(label_len)


def showByLabels(label):
    for i in range(len(dataset)):
        label = dataset.iloc[i, 2:6]
        if np.array_equal(label.values, label):
            img_name = dataset.iloc[i, 0]
            img_path = os.path.join(root_dir, img_name)
            img = cv2.imread(img_path)
            cv2.imshow('Zdjecie', img)

def countLabels():
    counters = np.zeros(8)
    for i in range(len(dataset)):
        label = dataset.iloc[i, 2:6]
        for j in range(len(possible_labels)):
            if Utils.hamming(label, possible_labels[j])==0:
                counters[j] += 1
                break
    print(possible_labels)
    print(counters)
    return counters

counters = countLabels()

for i in range(len(counters)):
    i_count = counters[i]
    if i_count < 300:
        for j in range(i_count, 300):
            pass
