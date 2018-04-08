# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import csv
import string
sample4=csv.reader(open('/home/lee/assignment3/sampleSubmission8.csv'))
sample6=csv.reader(open('/home/lee/assignment3/sampleSubmission1.csv'))
temp4=np.zeros([400,1])
temp6=np.zeros([400,1])
i=-1
num=0
num1=0
num2=0
for line4 in sample4:
    if line4[1]=='1':
        temp4[i][0]=1
        num1=num1+1
    if line4[1]=='0':
        temp4[i][0]=0
    i=i+1
print temp4

i=-1
for line6 in sample6:
    if line6[1]=='1':
        temp6[i][0]=1
        num2=num2+1
    if line6[1]=='0':
        temp6[i][0]=0
    i=i+1
print temp6
for j in range(len(temp4)):
    if temp4[j][0]==temp6[j][0]:
        num=num+1
print num1
print num2
print num
print round((num/400),2)
