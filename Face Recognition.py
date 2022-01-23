import time
import sys
import cv2
import numpy as np
import serial
from time import sleep
ser = serial.Serial('COM22',9600)  # open serial port

try_count = 10

start = time.time()
period = 8
face_cas = cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_default.xml')
vid_cam = cv2.VideoCapture(0,cv2.CAP_DSHOW)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Trainer.yml')
count = 0
flag = 0
Id = 2
filename = 'filename'
dict = {
    'item1': 1
}

 #Font = cv2.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 5, 1, 0, 1, 1)
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    _, img = vid_cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray, 1.3, 7)
    face_recog = False
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        Name, conf = recognizer.predict(roi_gray)
        print(Name, conf)
        if (Name <= 20 and conf < 40):
            if (Id == 2):
                face_recog = True
                break
        

        cv2.putText(img, str(Name) + " " + str(conf), (x, y - 10), font, 0.55, (120, 255, 120), 1)
           #cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,(0,0,255));
    cv2.imshow('frame', img)
        # cv2.imshow('gray',gray);
    if face_recog:
        print("Recognized")
    else:
        print('Unknown, can not recognize')
        flag = flag + 1

    count += 1
    if count >= try_count:
        break

    # if time.time() > start + period :
    #     break

    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

print("\n\n")
if flag >= try_count:
    print("No Face Recognized")
else:
    print("Face Recognized. Id =", Id)
    ser.write(b'on')     # write a string
    sleep(5)
    ser.write(b'off')     # write a string

vid_cam.release()
cv2.destroyAllWindows()
