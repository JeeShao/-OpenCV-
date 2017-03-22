# coding=utf-8
import cv2
# from ctypes import *
# import types

click= False
def onMouse(event,x,y,flags,param):
    global click
    if event==cv2.EVENT_LBUTTONUP:
        click=True
cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow('mywindow')
cv2.setMouseCallback('mywindow',onMouse)

print ('点击窗口或按任意键停止')
success,frame=cameraCapture.read()
while success and cv2.waitKey(1)==255 and not click:
    cv2.imshow('mywindow',frame)
    success,frame=cameraCapture.read()
cv2.destroyWindow('mywindow')
cameraCapture.release()
