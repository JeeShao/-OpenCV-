# -*- coding: utf-8 -*-
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *

from facerec import recognize, train
# from facerec import face
# from camera import VideoStream
# from configure import config, userManager

# from .soft_keyboard import *

import os
import cv2

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8


    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

# Qt css样式文件
# style = open('./ui/css/style.css').read()


# 显示视频的Qt控件
# setRect当前一帧图像上画出方框，用于标记人脸的位置
# setRectColor设置方框的颜色
# setUserLabel在方框旁边添加识别信息，比如识别到的用户名
class VideoFrame(QtGui.QLabel):
    userName = None
    pen_faceRect = QtGui.QPen()
    pen_faceRect.setColor(QtGui.QColor(255, 0, 0))
    x = 0;
    y = 0;
    w = 0;
    h = 0

    def __init__(self, parent):
        QtGui.QLabel.__init__(self, parent)

    def setRect(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    def setRectColor(self, r, g, b):
        self.pen_faceRect.setColor(QtGui.QColor(r, g, b))

    def setUserLabel(self, userName):
        self.userName = userName

    def paintEvent(self, event):
        QtGui.QLabel.paintEvent(self, event)
        painter = QtGui.QPainter(self)
        painter.setPen(self.pen_faceRect)
        painter.drawRect(self.x, self.y, self.w, self.h)
        if self.userName != None:
            painter.drawText(self.x, self.y + 15, self.userName)

# 人脸录入界面
class FaceRegister(QWidget):
    faceRect = None

    captureFlag = 0
    personName = None
    recOver = False
    model = None

    def __init__(self, mainWindow):
        super(FaceRegister, self).__init__()

        self.mainWindow = mainWindow
        # self.manager = userManager.UserManager()

        self.setupUi(self)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.playVideo)
        self._timer.start(10)

        self.update()

    def setupUi(self, FaceRegister):
        FaceRegister.setObjectName(_fromUtf8("FaceRegister"))
        FaceRegister.resize(400, 400)
        self.video_frame = VideoFrame(FaceRegister)
        self.video_frame.setGeometry(QtCore.QRect(0, 100, 480, 360))
        # QtCore.QMetaObject.connectSlotsByName(FaceRegister)

    def setVideo(self, video):
        self.video = video
        if self.video.is_release:
            self.video.open(0)

    def playVideo(self):
        try:
            pixMap_frame = QtGui.QPixmap.fromImage(self.video.getQImageFrame())
            train.detect()
            self.video_frame.setPixmap(pixMap_frame)
            self.video_frame.setScaledContents(True)
        except TypeError:
            print('No frame')

