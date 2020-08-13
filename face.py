#!/usr/bin/env python
# coding: utf-8

# In[1]:



import cv2; import numpy as np
"""
this face detection code
"""


frameWidth= 640
frameHeight= 480
cap= cv2.VideoCapture(0)  #0 for the first (or only) camera     1 for the second/rear camera when found
cap.set(3,frameWidth)   #id 3 for the width --> width=640
cap.set(4,frameHeight)  # id 4 for the height--> height=480
cap.set(10,150) #id-10 for the brightness--> brightness increases 

while True:    #
    success, frame=cap.read()   #success is boolean variable (True or False)    # cap== Capture frame-by-frame
    imgGray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#find the faces
    faceCascade= cv2.CascadeClassifier('C:/Users/Abouzid/Google Drive (engineer.m90@gmail.com)/training- opencv/resources/haarcascade_frontalface_default.xml')  #haar cascade (trained for face detection) # i can create one for detecting whatever I want
    faces= faceCascade.detectMultiScale(imgGray, 1.1, 4)   #(img, scale factor, min neighbour)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w,y+h), (255,0,0), 2)
      
    
    
    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF ==ord('q'):   #it enables: press q button to close
        break
        


# In[ ]:




