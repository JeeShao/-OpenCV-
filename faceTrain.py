#!/usr/bin/env python
# -- coding:utf-8 --
#@Time  : 2017/3/22  
#@Author: Jee
import numpy as np
import cv2

def detect():
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
    camera = cv2.VideoCapture(0)
    count=1
    while(True):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5,cv2.CASCADE_DO_CANNY_PRUNING)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            f = cv2.resize(gray[y:y+h,x:x+w],(92,112))
            if(cv2.waitKey(100) & 0xff == ord("m")):  #拍照保存
                cv2.imwrite('./data/s41/%s.pgm'% str(count),f)
                count+=1
            roi_color = img[y:y+h,x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_color,1.03,5,0,(40,40))
            for(ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.imshow("camera",frame)
        if cv2.waitKey(1000//12) & 0xff == ord("q") or count==11:
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect()