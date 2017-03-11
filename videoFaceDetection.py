#!/usr/bin/env python
# -- coding:utf-8 --
#@Time  : 2017/3/11  
#@Author: Jee
import cv2

def detect():
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
    camera = cv2.VideoCapture(0)
    while(True):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        # print "Faces:",faces
        for (x, y, w, h) in faces:
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_color = img[y:y+h,x:x+w]
            # print "GRAY",roi_gray;
            # print "COLOR",roi_color;
            eyes = eye_cascade.detectMultiScale(roi_gray,1.1,5,0,(40,40))
            # eyes1 = eye_cascade.detectMultiScale(roi_color,1.2,5,0,(40,40))
            # print "GrayEye:",eyes;
            # print "ColorEye:",eyes1;
            for(ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        cv2.imshow("camera",frame)
        if cv2.waitKey(1000/12) & 0xff == ord("q"):
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect()