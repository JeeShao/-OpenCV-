#!/usr/bin/env python
# -- coding:utf-8 --
#@Time  : 2017/3/22  
#@Author: Jee
import numpy as np
import cv2
import os
import sys
from doCsv import doCsv

def detect(path,dir=''):
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier('./cascades/haarcascade_mcs_mouth.xml')
    nose_cascade = cv2.CascadeClassifier('./cascades/haarcascade_mcs_nose.xml')
    camera = cv2.VideoCapture(0)
    count=1
    if(dir==''):
        # 增加新人脸时新建目录
        dirnames=os.listdir(path)
        dirnames.sort(key=lambda x:int(x.split('s')[1]),reverse=True)
        dir_no=int(dirnames[0].split('s')[1])+1
        dir = "s"+str(dir_no)
    if(os.path.exists('./data/%s'% dir)==False): #目录不存在则创建
        os.mkdir('./data/%s'% dir)
    while(True):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#灰度图像
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        k = cv2.waitKey(1000 // 12)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            f = cv2.resize(gray[y:y+h,x:x+w],(92,112))#格式化大小
            if(k & 0xff == ord("m")):  #拍照保存
                f = cv2.equalizeHist(f)  # 均衡直方图
                cv2.imwrite('./data/%s/%s.pgm'% (dir,str(count)),f)
                print("拍照:%d"%(count))
                count+=1
            roi_color = img[y:y+h,x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_color,1.3,5,0,(40,40))
            for(ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            mouth = mouth_cascade.detectMultiScale(roi_color,1.3,40,0,(40,40))
            for (ex, ey, ew, eh) in mouth:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,100,0),2)
            nose = nose_cascade.detectMultiScale(roi_color,1.3,20,0,(30,30))
            for (ex, ey, ew, eh) in nose:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(100,100,0),2)
        cv2.imshow("camera",frame)
        if k & 0xff == ord("q") or count==51:#拍照60张或按下q键 则退出
            break
    camera.release()
    cv2.destroyAllWindows()

def imagestoCsv(path,sz=None):
    data = []
    c=0
    sort_flag=0
    for dirname,dirnames,filenames in os.walk(path):
        if sort_flag==0:
            dirnames.sort(key=lambda x: int(x.split('s')[1]))
            sort_flag=1      #标志排过序 以免重复排序
        for subdirname in dirnames:
            subject_path = os.path.join(dirname,subdirname)
            for filename in os.listdir(subject_path):
                try:
                    if(filename == ".directory"):
                        continue
                    filepath = os.path.join(subject_path,filename)
                    # im = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
                    data = data + [(filepath,str(c))]
                    # 调整尺寸
                    # if(sz is not None):
                    #     im = cv2.resize(im,(92,112))
                    # X.append(np.asarray(im,dtype=np.uint8))
                    # y.append(c)
                except IOError as xxx_todo_changeme:
                    (errno,strerror) = xxx_todo_changeme.args
                    print("I/O error({0}):{1}".format(errno,strerror))
                except:
                    print("Unexpected error:",sys.exc_info()[0])
                    raise
            c = c+1
    docsv.csv_writer(data)
if __name__ == "__main__":
    path = "./data"
    dir = 's0'  #保存训练人脸图的目录，为空表示新建
    docsv = doCsv("trainFace.csv")
    detect(path,dir)
    imagestoCsv(path)
