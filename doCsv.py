#!/usr/bin/env python
# coding: utf-8
import csv
# import datetime
import time
# import codecs
class doCsv:
    def __init__(self,file= "csv_test.csv"):
        self.file=file
        # self.data=''

    def csv_writer(self,data):
        try:
            with open(self.file, 'w', newline='') as csvfile:
                # csvfile.write(codecs.BOM_UTF8)
                now = time.strftime("%Y-%m-%d %H:%M:%S")
                writer = csv.writer(csvfile, dialect='excel')
                writer.writerow([now])
                writer.writerows(data)
        except:
            print("csv writer error")
            raise

    def csv_reader(self):
        resList = []
        try:
            with open(self.file, 'r',) as csvfile:
                reader = csv.reader(csvfile)
                for line in reader:
                    resList = resList + line
            if resList:
                return resList
            else:
                return False
        except:
            print("csv reader error")
            raise

