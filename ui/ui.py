#!/usr/bin/env python
# -- coding:utf-8 --
#@Time  : 2017/4/5  
#@Author: Jee

from __future__ import division
import sys
from math import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import PyQt5
class Form(QDialog):
    def __init__(self,parent):
        dial=QDial()
        dial.setNot