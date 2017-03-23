#!/usr/bin/env python
# -- coding:utf-8 --
#@Time  : 2017/3/22  
#@Author: Jee
import numpy as np
import cv2
from videoFaceDetection import imagestoCsv
import os

def detect(path):
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_alt2.xml')
    eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier('./cascades/haarcascade_mouth.xml')
    nose_cascade = cv2.CascadeClassifier('./cascades/haarcascade_nose.xml')
    camera = cv2.VideoCapture(0)
    count=1
    dir = 's42'
    if(os.path.exists('./data/%s'% dir)==False): #目录不存在则创建
        os.mkdir('./data/%s'% dir)
    while(True):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#灰度
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        k = cv2.waitKey(1000 // 12)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            f = cv2.resize(gray[y:y+h,x:x+w],(92,112))#格式化大小
            if(k & 0xff == ord("m")):  #拍照保存
                cv2.imwrite('./data/%s/%s.pgm'% (dir,str(count)),f)
                print("拍照:%d"%(count))
                count+=1
            roi_color = img[y:y+h,x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_color,1.3,8,0,(40,40))
            for(ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            mouth = mouth_cascade.detectMultiScale(roi_color,1.3,40,0,(40,40))
            for (ex, ey, ew, eh) in mouth:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,100,0),2)
            nose = nose_cascade.detectMultiScale(roi_color,1.3,20,0,(30,30))
            for (ex, ey, ew, eh) in nose:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(100,100,0),2)
        cv2.imshow("camera",frame)
        if k & 0xff == ord("q") or count==41:
            break
    camera.release()
    cv2.destroyAllWindows()
    imagestoCsv(path)

if __name__ == "__main__":
    path = "./data"
    detect(path)