"""
Loading, preprocessing, and formatting data.
"""

import os
import re
import csv
import cv2
import numpy as np
from random import randint

import settings

class Data:
    def __init__(self):
        self.video_files = []
        self.label_files = []

        self.videos = []
        self.labels = []
        
        self.shuffled_videos = []
        self.shuffled_labels = []
        
        self.episodes = []

        self.cwd = os.getcwd()
        self.data_path = os.path.join(self.cwd, "data")

    def load_files(self):
        files = os.listdir(self.data_path)
        files.remove(".gitkeep")
        for file in files:
            if re.match(".*\.avi", file):
                self.video_files.append(file)
            elif re.match(".*\.csv", file):
                self.label_files.append(file)
            else:
                raise TypeError("File name didn't match any patterns:", file)

    def process_data(self):
        for i, file in enumerate(self.label_files):
            self.labels.append([])
            with open(os.path.join(self.data_path, file)) as csvfile:
                reader = csv.reader(csvfile)
                for j, row in enumerate(reader):
                    if j:
                        self.labels[i].append(float(row[1]))
            self.labels[i] = np.array(self.labels[i])

        for i, file in enumerate(self.video_files):
            self.videos.append([])
            video = cv2.VideoCapture(os.path.join(self.data_path, file))

            for j in range(len(self.labels[i])):
                retval, img = video.read()
                self.videos[i].append(self.preprocess(img))
            self.videos[i] = np.array(self.videos[i])
        self.episodes = list(zip(*self.get_data()))
		 
    def get_data(self):
        return self.videos, self.labels
        
    def get_episodes(self):
        return self.episodes
        
    def get_collapsed_data(self, episodes):
        collapsed_videos = []
        collapsed_labels = []
        
        zipped_episodes = [list(zip(video, labels)) for video, labels in episodes]
        for episode in zipped_episodes:
            for frame, label in episode:
                collapsed_videos.append(frame)
                collapsed_labels.append(label)
        
        return np.array(collapsed_videos), np.array(collapsed_labels)
        
    def get_shuffled_data(self, videos, labels):
        n = sum([len(video) for video in videos])
        m = len(videos[0])
        indices = list(range(n))
        self.shuffled_videos = [0 for i in range(n)]
        self.shuffled_labels = [0 for i in range(n)]
        for i in range(n):        
            rand_index = randint(0, len(indices)-1)
            j = indices.pop(rand_index)
            self.shuffled_videos[i] = videos[j//m][j%m]
            self.shuffled_labels[i] = labels[j//m][j%m]
        self.shuffled_videos = np.array(self.shuffled_videos)
        self.shuffled_labels = np.array(self.shuffled_labels)
	
        return self.shuffled_videos, self.shuffled_labels
          
    def preprocess(self, img):  # uncommnet comments to view before and after
        #self.disp_img(img)
        img = cv2.resize(img, settings.img_size_rev)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255
        #self.disp_img(img)
        return img
        
    def disp_img(self, img):
        cv2.imshow('image', img)
        cv2.waitKey(0)
