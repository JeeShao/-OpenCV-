#!/usr/bin/env python
# -- coding:utf-8 --
#@Time  : 2017/3/22  
#@Author: Jee
import numpy as np
import cv2
import sys
import os

def detect():
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
    camera = cv2.VideoCapture(0)
    count=1
    while(True):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            f = cv2.resize(gray[y:y+h,x:x+w],(92,112))
            if(cv2.waitKey(1000//12) & 0xff == ord("m")):
                cv2.imwrite('./data/s41/%s.pgm'% str(count),f)
                count+=1
            # roi_gray = gray[y:y+h,x:x+w]
            # roi_color = img[y:y+h,x:x+w]
            # eyes = eye_cascade.detectMultiScale(roi_gray ,1.03,5,0,(40,40))
            # # eyes1 = eye_cascade.detectMultiScale(roi_color,1.2,5,0,(40,40))
            # # print "GrayEye:",eyes;
            # # print "ColorEye:",eyes1;
            # for(ex,ey,ew,eh) in eyes:
            #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.imshow("camera",frame)
        if cv2.waitKey(1000//12) & 0xff == ord("q") or count==11:
            break
    camera.release()
    cv2.destroyAllWindows()

# 加载数据并识别人脸
def read_images(path, sz=None):
    c=1
    X,y = [],[]
    for dirname,dirnames,filenames in os.walk(path):
        for subdirname in dirnames:
            # print(c,subdirname)
            subject_path = os.path.join(dirname,subdirname)
            for filename in os.listdir(subject_path):
                try:
                    if(filename == ".directory"):
                        continue
                    filepath = os.path.join(subject_path,filename)
                    im = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
                    # 调整尺寸
                    if(sz is not None):
                        im = cv2.resize(im,(92,112))
                    X.append(np.asarray(im,dtype=np.uint8))
                    y.append(c)
                except IOError as xxx_todo_changeme:
                    (errno,strerror) = xxx_todo_changeme.args
                    print("I/O error({0}):{1}".format(errno,strerror))
                except:
                    print("Unexpected error:",sys.exc_info()[0])
                    raise
            c = c+1
    return [X,y]

# 基于Eigenfaces算法测试人脸识别脚本
def face_rec():
    names = ["Jee",'Joe','Jack']
    sysargv={0:'00',1:'./data',2:'./train'}
    if len(sysargv)<2:
        print("USAGE:facerec_demo.py </path/to/image> [</path/to/store/image/at>]")
        sys.exit()
    [X,y] = read_images(sysargv[1])
    y = np.asarray(y,dtype=np.int32)
    if(len(sysargv) == 3):
        out_dir = sysargv[2]
    model = cv2.face.createEigenFaceRecognizer()
    model.train(np.asarray(X),np.asarray(y))
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    while(True):
        read,img = camera.read()
        # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        print(faces)
        # exit()
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            roi = gray[x:x+w,y:y+h]
            try:
                roi = cv2.resize(roi, (92, 112),interpolation=cv2.INTER_LINEAR)
                params = model.predict(roi)
                print("Lable: %s, Confidence: %.2f" % (params[0],params[1]))
                cv2.putText(img,str(params[0]),(x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)
            except:
                continue
        cv2.namedWindow("camera",cv2.WINDOW_NORMAL)
        if(img is not None):
            cv2.imshow("camera",img)
        if cv2.waitKey(1000 // 12) & 0xff == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()
flag = 1 #0:训练 1::识别
if __name__ == "__main__":
    face_rec() if flag else detect()