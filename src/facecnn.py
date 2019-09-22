import cv2
import sys

cascPath = "../resources/models/haarcascade_frontalface_default.xml"
faceCasc = cv2.CascadeClassifier(cascPath)

video = cv2.VideoCapture(0)

while True:
    r, f = video.read()
    
    g = cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
    faces = faceCasc.detectMultiScale(
        g, 
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (28,28),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    
    for (x,y,w,h) in faces:
        cv2.rectangle(f,(x,y), (x+w,y+h), (0,255,0), 2)
    
    cv2.imshow('Video',f)
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break;
        
video.release()
cv2.destroyAllWindows()