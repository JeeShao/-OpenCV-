#!/usr/bin/env python
# coding: utf-8
import csv
# import codecs
class doCsv:
    def __init__(self,file= "csv_test.csv"):
        self.file=file
        # self.data=''

    def csv_writer(self,data):
        with open(self.file, 'a', newline='') as csvfile:
            # csvfile.write(codecs.BOM_UTF8)
            writer = csv.writer(csvfile, dialect='excel')
            writer.writerows(data)

    def csv_reader(self):
        with open(self.file, 'r',) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                print(line)

