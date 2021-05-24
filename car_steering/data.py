"""
Contains code to load, preprocess, and store the training data.
"""

import os
import re
import csv
import cv2
import numpy as np
from random import randint

import settings

class Data:
    """
    Access point to training data.
    """
    def __init__(self) -> None:
        self.video_files = []  # file names with .avi extension corresponding to training frames
        self.label_files = []  # file names with .csv extension corresponding to training labels

        self.videos = []  # np.array(list[list[np.array]]), contains sub list for each episode with that episode's frames
        self.labels = []  # np.array(list[list[float]]), contains sub list for each episode with that episode's labels
        
        self.shuffled_videos = []  # contains randomly shuffles flattened video frames
        self.shuffled_labels = []  # # contains randomly shuffles flattened labels
        
        self.episodes = []  # zipped lsit of video frames and labels for each episode

        self.cwd = os.getcwd()  # current working directory
        self.data_path = os.path.join(self.cwd, "data")  # path to directory for training data

    def load_files(self) -> None:
        """
        Find and store names of files containing training data.
        
        Exceptions:
            Throws TypeError if finds file not matching expected patterns.
        """
        files = os.listdir(self.data_path)
        files.remove(".gitkeep")
        for file in files:
            if re.match(".*\.avi", file):
                self.video_files.append(file)
            elif re.match(".*\.csv", file):
                self.label_files.append(file)
            else:
                raise TypeError("File name didn't match any patterns:", file)

    def process_data(self) -> None:
        for i, file in enumerate(self.label_files):  # gettings labels for each training episode
            self.labels.append([])
            with open(os.path.join(self.data_path, file)) as csvfile:
                reader = csv.reader(csvfile)
                for j, row in enumerate(reader):
                    if j:  # if not empty
                        self.labels[i].append(float(row[1]))  # add label to list
            self.labels[i] = np.array(self.labels[i])  # convert list to numpy array

        for i, file in enumerate(self.video_files):  # gettings videos for each training episode
            self.videos.append([])
            video = cv2.VideoCapture(os.path.join(self.data_path, file))

            for j in range(len(self.labels[i])):  # iterate through each frame in video
                retval, img = video.read()
                self.videos[i].append(self.preprocess(img))  # add preprocessed frame
            self.videos[i] = np.array(self.videos[i])  # convert to numpy array

        self.episodes = list(zip(*self.get_data()))  # create zipped list for each episode
         
    def get_data(self) -> tuple:
        """
        Returns: unzipped videos and labels for each episode
        """
        return self.videos, self.labels
        
    def get_episodes(self) -> list:
        """
        Returns: zipped videos and labels for each episode
        """    
        return self.episodes
        
    def get_collapsed_data(self, episodes: np.ndarray) -> tuple:
        """
        'Collapses' or flattens several episodes of data together.
        
        Returns: collapsed data for given episodes.
        """
        collapsed_videos = []
        collapsed_labels = []
        
        zipped_episodes = [list(zip(video, labels)) for video, labels in episodes]
        for episode in zipped_episodes:
            for frame, label in episode:
                collapsed_videos.append(frame)
                collapsed_labels.append(label)
        
        return np.array(collapsed_videos), np.array(collapsed_labels)
        
    def get_shuffled_data(self, videos: np.ndarray, labels: np.ndarray) -> tuple:
        """
        Randomly shuffles and returns data.
        """
        n = sum([len(video) for video in videos])
        m = len(videos[0])
        indices = list(range(n))  # all possible indices that need to be addressed
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
          
    def preprocess(self, img: np.ndarray) -> np.ndarray:  # uncomment comments to view before and after preprocessing
        #self.disp_img(img)
        img = cv2.resize(img, settings.img_size_rev)  # resize based on target size determined by settings.py
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to black and white
        img = img / 255  # normalize values
        #self.disp_img(img)
        return img
        
    def disp_img(self, img: np.ndarray) -> None:
        cv2.imshow('image', img)
        cv2.waitKey(0)

