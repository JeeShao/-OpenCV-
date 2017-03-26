#!/usr/bin/env python
# -- coding:utf-8 --
#@Time  : 2017/3/22  
#@Author: Jee
import numpy as np
import cv2
import os
import sys
from doCsv import doCsv
import config

def detect(path,dir=''):
    face_cascade = cv2.CascadeClassifier(config.FACE_CLASSIFIER_FILE)
    eye_cascade = cv2.CascadeClassifier(config.EYES_CLASSIFIER_FILE)
    mouth_cascade = cv2.CascadeClassifier(config.MOUTH_CLASSIFIER_FILE)
    nose_cascade = cv2.CascadeClassifier(config.NOSE_CLASSIFIER_FILE)
    camera = cv2.VideoCapture(0)
    count=1
    if(dir==''):
        # 增加新人脸时新建目录
        dirnames=os.listdir(path)
        dirnames.sort(key=lambda x:int(x.split('s')[1]),reverse=True)
        dir_no=int(dirnames[0].split('s')[1])+1
        dir = "s"+str(dir_no)
    if(os.path.exists(config.TRAINING_DIR+'/%s'% dir)==False): #目录不存在则创建
        os.mkdir(config.TRAINING_DIR+'/%s'% dir)
    while(True):
        ret, frame = camera.read()
        if ret:
            frame = cv2.flip(frame,1)  #镜像翻转
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#灰度图像
        faces = face_cascade.detectMultiScale(gray,config.HAAR_SCALE_FACTOR,config.HAAR_MIN_NEIGHBORS,0,config.HAAR_MIN_SIZE)
        k = cv2.waitKey(1000 // 12)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            f = cv2.resize(gray[y:y+h,x:x+w],(config.FACE_WIDTH, config.FACE_HEIGHT))#格式化大小
            if(k & 0xff == ord("m")):  #拍照保存
                f = cv2.equalizeHist(f)  # 均衡直方图
                cv2.imwrite(config.TRAINING_DIR+'/%s/%s.pgm'% (dir,str(count)),f)
                print("拍照:%d"%(count))
                count+=1
            roi_color = img[y:y+h,x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_color,config.HAAR_SCALE_FACTOR,8,0,config.HAAR_MIN_SIZE)
            draw_rects(roi_color,eyes,(0,255,0))

            mouth = mouth_cascade.detectMultiScale(roi_color,config.HAAR_SCALE_FACTOR,40,0,config.HAAR_MIN_SIZE)
            draw_rects(roi_color,mouth,(0,100,0))

            nose = nose_cascade.detectMultiScale(roi_color,config.HAAR_SCALE_FACTOR,10,0,config.HAAR_MIN_SIZE)
            draw_rects(roi_color,nose,(100,100,0))
        cv2.imshow("camera",frame)
        if k & 0xff == ord("q") or count==101:#拍照60张或按下q键 则退出
            break
    camera.release()
    cv2.destroyAllWindows()

#绘制矩形框
def draw_rects(img, rects, color):
    for x, y, w, h in rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

def imagestoCsv(path,sz=None):
    data = []
    label=1
    sort_flag=0
    for dirname,dirnames,filenames in os.walk(path):
        if sort_flag==0:
            dirnames.sort(key=lambda x: int(x.split('s')[1]))
            sort_flag=1      #标志排过序 以免重复排序
        for subdirname in dirnames:
            # subject_path = "%s/%s" % (dirname, subdirname)
            subject_path = os.path.join(dirname,subdirname).replace('\\', '/')
            for filename in os.listdir(subject_path):
                try:
                    if(filename == ".directory"):
                        continue
                    filepath = os.path.join(subject_path,filename).replace('\\', '/')
                    # im = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
                    # data1 = data1 + [(filepath,str(label))]
                    data = data + [tuple(["%s;%d" % (filepath,label)])]
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
            label = label+1
    docsv.csv_writer(data)

def saveModel():
    X, y = [], []
    img_list = docsv.csv_reader()
    if img_list:
        for i in range(1, len(img_list)):
            img_str = img_list[i].split(';')
            filepath = img_str[0]
            try:
                im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                c = img_str[1]
            except:
                traceback.print_exc()
            # 调整尺寸
            # if (sz is not None):
            #     im = cv2.resize(im, (92, 112))
            X.append(np.asarray(im, dtype=np.uint8))
            y.append(c)
        y = np.asarray(y, dtype=np.int32)
        model = cv2.face.createLBPHFaceRecognizer()
        model.train(np.asarray(X), np.asarray(y))
        model.save(config.TRAINING_MODEL)
    else:
        return False

if __name__ == "__main__":
    path = config.TRAINING_DIR
    dir = 's41'  #保存训练人脸图的目录，为空表示新建
    docsv = doCsv(config.TRAINING_CVS_FILE)
    detect(path,dir)
    imagestoCsv(path)
    saveModel()
