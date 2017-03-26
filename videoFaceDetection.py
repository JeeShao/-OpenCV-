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
import config
import time

# 基于LBPHfaces算法测试人脸识别脚本
def face_rec():
    model = cv2.face.createLBPHFaceRecognizer()
    t0 = time.time()
    model.load(config.TRAINING_MODEL)
    t1 = time.time()
    print("加载训练模型耗时",t1-t0,'S')
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(config.FACE_CLASSIFIER_FILE)
    while(True):
        read,img = camera.read()
        img = cv2.flip(img,1) #镜像翻转
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
                roi = cv2.resize(roi, (config.FACE_WIDTH, config.FACE_HEIGHT),interpolation=cv2.INTER_LINEAR)#格式化
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

if __name__ == "__main__":
    path = "./data"
    face_rec()