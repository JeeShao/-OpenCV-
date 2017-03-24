#!/usr/bin/env python
# -- coding:utf-8 --
#@Time  : 2017/3/22  
#@Author: Jee
import numpy as np
import cv2
import sys
import os
import traceback
from doCsv import doCsv

# 加载数据并识别人脸


def readImages(sz=None):
    X, y = [], []
    docsv = doCsv("trainFace.csv")
    img_list = docsv.csv_reader()
    if img_list:
        for i in range(1,len(img_list)):
            img_str = img_list[i].split(';')
            filepath = img_str[0]
            try:
                im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                c = img_str[1]
            except:
                traceback.print_exc()
            # 调整尺寸
            if (sz is not None):
                im = cv2.resize(im, (92, 112))
            X.append(np.asarray(im, dtype=np.uint8))
            y.append(c)
        return [X,y]
    else:
        return False

# 基于Eigenfaces算法测试人脸识别脚本
def face_rec():
    names = ["Jee",'Joe','Jack']
    sysargv={0:'00',1:'./data',2:'./train'}
    if len(sysargv)<2:
        print("USAGE:facerec_demo.py </path/to/image> [</path/to/store/image/at>]")
        sys.exit()
    try:
        [X,y] = readImages()
    except:
        print("read images error")
        traceback.print_exc()
        exit(1)
    y = np.asarray(y,dtype=np.int32)
    if(len(sysargv) == 3):
        out_dir = sysargv[2]
    model = cv2.face.createLBPHFaceRecognizer()
    model.train(np.asarray(X),np.asarray(y))
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_alt.xml')
    while(True):
        read,img = camera.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #灰度化
        #分离彩色图三通道
        # b, g, r = cv2.split(frame)
        # # cv2.imshow("Blue", r)
        # 直方图均衡化
        # hist_b=cv2.equalizeHist(b)
        # hist_g=cv2.equalizeHist(g)
        # hist_r=cv2.equalizeHist(r)
        #合并三通道(顺序是bgr)
        # img = cv2.merge(([hist_b, hist_g, hist_r]))
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print("faces",faces)
        for (x, y, w, h) in faces: #多张人脸时 循环识别
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            roi = gray[x:x+w,y:y+h] #灰度人脸图
            try:
                roi = cv2.resize(roi, (92, 112),interpolation=cv2.INTER_LINEAR)#格式化
                hist_roi = cv2.equalizeHist(roi)  # 均衡直方图
                params = model.predict(hist_roi)
                print("Lable: %s, Confidence: %.2f" % (params[0],params[1]))
                p=list(params)
                if p[0]==43:
                    p[0]="SGX"
                elif p[0]==41:
                    p[0]="SJ"
                elif p[0]==42:
                    p[0]="LC"
                else:
                    pass
                cv2.putText(img,str(p[0]),(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)
            except:
                continue
        cv2.namedWindow("camera",cv2.WINDOW_NORMAL)
        if(img is not None):
            cv2.imshow("camera",img)
        if cv2.waitKey(1000 // 12) & 0xff == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()

flag = 1 #0:训练 1:识别
if __name__ == "__main__":
    path = "./data"
    face_rec() if flag else detect(path)