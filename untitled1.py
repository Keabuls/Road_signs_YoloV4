# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 14:14:35 2023

@author: asonm
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    frame_blob = cv2.dnn.blobFromImage(frame,1/255,(1920,1080),swapRB=True,crop=False)
    
    
    labels = ["DUR","Yaya Geçidi","Yol ver"]
    
    colors = ["0,255,255","0,0,255","255,255,0","0,255,0","255,0,0","255,0,255"]
    colors = [np.array(color.split(",")).astype("int") for color in colors]
    colors = np.array(colors)
    colors = np.tile(colors,(18,1))

    model = cv2.dnn.readNetFromDarknet()