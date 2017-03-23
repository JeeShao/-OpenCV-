#!/usr/bin/env python
# -- coding:utf-8 --
#@Time  : 2017/3/11
#@Author: Jee
import os
import sys
import cv2
import numpy as np
# 加载数据并识别人脸
path ='./data'
def read_images(path, sz=None):
    c=0
    X,y = [],[]
    for dirname,dirnames,filenames in os.walk(path):
        for filename in os.listdir(dirname):
            try:
                if(filename == ".directory"):
                    continue
                filepath = os.path.join(dirname,filename)
                im = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
                # 调整尺寸
                if(sz is not None):
                    im = cv2.resize(im,(200,200))
                X.append(np.asarray(im,dtype=np.uint8))
                y.append(c)
            except IOError as xxx_todo_changeme:
                (errno,strerror) = xxx_todo_changeme.args
                print("I/O error({0}):{1}".format(errno,strerror))
            except:
                print("Unexpected error:",sys.exc_info()[0])
                raise
        c = c+1
    print(X)

# Form implementation generated from reading ui file 'demo2.py'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

