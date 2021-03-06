#coding=utf-8
#静态图片的人脸检测
#2017.03.11
#Jee
import cv2
filename = '1.jpg'

def detect(filename):
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    img = cv2.imread(filename)
    # print img
    # exit()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.1,5)
    for(x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.namedWindow('MyFace',0)
    cv2.imshow('MyFace',img)
    cv2.imwrite('./MyFace1.jpg',img)
    cv2.waitKey(0)
detect(filename)