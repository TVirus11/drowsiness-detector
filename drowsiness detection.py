import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

mixer.init() 
sound = mixer.Sound('alarm.wav')

# xml files for detecting face, left eye and right eye
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')


label =['Close','Open']

model = load_model('models/model4.h5') # loading the model .
path = os.getcwd()

cap = cv2.VideoCapture(0) # Capturing the video through the camera

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thick=2
rpred=[99]
lpred=[99]

# the loop for performing all the operations at real time
while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    # Right Eye Detection
    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 1)   # Rectangle around the right eye
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict_classes(r_eye)
        if(rpred[0]==1):
            label ='Open'
        if(rpred[0]==0):
            label ='Closed'
        break
    
    # Left Eye Detection
    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 1)    # Rectangle around the left eye
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        if(lpred[0]==1):
            label ='Open'
        if(lpred[0]==0):
            label ='Closed'
        break
        
    
    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
    if(score<0):
        score=0   
    cv2.putText(frame,'Time:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
    if(score>5):
        try:
            sound.play()     #playing the alarm when time is above 5 sec.
        except:
            pass
        
        if(thick<16):
            thick= thick+2
        else:
            thick=thick-2
            if(thick<2):
                thick=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thick)
        
    if (score == 4):       # Stop the alarm
        sound.stop()
    
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
cap.release()
cv2.destroyAllWindows()
