#!/usr/bin/env python
# -- coding:utf-8 --
#@Time  : 2017/4/7  
#@Author: Jee

import sys
import cv2

from ui import mainHome
from camera import Video
import config
from PyQt4.QtGui import QApplication


def main():
    # model = cv2.face.createLBPHFaceRecognizer()
    # model.load(config.TRAINING_FILE)

    video = Video.Video(0)
    video.setFrameSize(640, 480)
    video.setFPS(30)

    QtApp = QApplication(sys.argv)

    mainWindow = mainHome.Ui_MainWindow()
    # mainWindow.setModel(model)
    mainWindow.show()
    mainWindow.setVideo(video)
    mainWindow.raise_()

    QtApp.exec_()


if __name__ == '__main__':
    main()
