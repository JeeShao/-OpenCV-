import cv2
from PyQt4.QtGui import QImage
import numpy as np
from PyQt4.Qt import QSize, qRgb

class Ipl2QImage(QImage):
    def __init__(self, iplimage, QFormat=QImage.Format_RGB888):#这里设置了默认的Format
        if iplimage.channels ==3 and QFormat==QImage.Format_RGB888:#如果是默认彩色情况
            super(Ipl2QImage,self).__init__(iplimage.tostring(), iplimage.width, iplimage.height, QFormat)
        elif iplimage.channels == 1 and QFormat==QImage.Format_RGB888:#如果是默认单色情况
            super(Ipl2QImage,self).__init__(iplimage.tostring(), iplimage.width, iplimage.height, QImage.Format_Indexed8)
        else:#其他情况，这时当然就需要自己传入Format了
            super(Ipl2QImage,self).__init__(iplimage.tostring(), iplimage.width, iplimage.height, QFormat)

if __name__ == '__main__':
    img = cv2.imread('1.jpg')
    a=Ipl2QImage(img)

