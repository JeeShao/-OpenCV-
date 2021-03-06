#!/usr/bin/env python
# -- coding:utf-8 --
#@Time  : 2017/4/6  
#@Author: Jee
import datetime
import sys
from time import strftime

from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from ui import ui

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

# style = open('./ui/css/style.css').read()


class Ui_MainWindow(QMainWindow):
    model = None
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        # self.setStyleSheet(style)

    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(400, 400)

        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout = QtGui.QVBoxLayout(self.centralwidget)

        self.label_welcome = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(28)
        self.label_welcome.setFont(font)
        self.label_welcome.setAlignment(QtCore.Qt.AlignCenter)
        self.verticalLayout.addWidget(self.label_welcome)

        self.gridLayout_buttons = QtGui.QGridLayout()
        self.gridLayout_buttons.setSizeConstraint(QtGui.QLayout.SetMaximumSize)
        self.gridLayout_buttons.setMargin(0)
        self.gridLayout_buttons.setSpacing(50)

        # button face
        self.btn_face = QtGui.QPushButton(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_face.sizePolicy().hasHeightForWidth())
        self.btn_face.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(26)
        self.btn_face.setFont(font)
        self.gridLayout_buttons.addWidget(self.btn_face, 0, 0, 1, 1)

        self.btn_face.clicked.connect(self.btn_face_clicked)

        # button register
        self.btn_register = QtGui.QPushButton(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_register.sizePolicy().hasHeightForWidth())
        self.btn_register.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(26)
        self.btn_register.setFont(font)
        self.gridLayout_buttons.addWidget(self.btn_register, 1, 0, 1, 1)

        self.btn_register.clicked.connect(self.btn_register_clicked)

        self.verticalLayout.addLayout(self.gridLayout_buttons)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 2)
        self.verticalLayout.setStretch(2, 1)
        self.verticalLayout.setStretch(3, 4)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 480, 23))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def setModel(self, model):
        self.model = model

    def setVideo(self, video):
        self.video = video

    def btn_face_clicked(self):
        print('facerec clicked')
        self.video.open(0)
        self._timer.stop()

        self.facerec = ui.FaceRec(self)
        self.facerec.setModel(self.model)
        self.facerec.setVideo(self.video)
        self.setCentralWidget(self.facerec)

    def btn_register_clicked(self):
        self.register = ui.FaceRegister(self)
        self.register.setVideo(self.video)
        self.setCentralWidget(self.register)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.label_welcome.setText(str("人脸识别系统"))
        self.btn_face.setText("人脸识别")
        self.btn_register.setText("人脸录入")

if __name__ == "__main__":
    QtApp = QApplication(sys.argv)

    mainWindow = Ui_MainWindow()
    # mainWindow.setModel(model)
    # mainWindow.showFullScreen()
    # mainWindow.setVideo(video)
    mainWindow.show()
    mainWindow.raise_()

    QtApp.exec_()

