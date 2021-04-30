"""

"""

import os
import re
import csv
import cv2

import settings

video_files = []
label_files = []

videos = []
labels = []

cwd = os.getcwd()
data_path = os.path.join(cwd, "car_data")

def get_files():
    files = os.listdir(data_path)
    files.remove(".gitkeep")
    for file in files:
        if re.match(".*\.avi", file):
            video_files.append(file)
        elif re.match(".*\.csv", file):
            label_files.append(file)
        else:
            raise TypeError("File name didn't match any patterns...")

def get_raw_data():
    for i, file in enumerate(label_files):
        labels.append([])
        with open(os.path.join(data_path, file)) as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                labels.append(row[1])
    for i, file in enumerate(video_files):
        videos.append([])
        video = cv2.VideoCapture(os.path.join(data_path, file))
        for j in range(len(labels[i])):
            retval, img = video.read()
            videos[i].append(preprocess(img))
            
def preprocess(img):
    img = cv2.resize(img, (settings.img_height, settings.img_width))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255
    return img
    
def disp_img(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)

