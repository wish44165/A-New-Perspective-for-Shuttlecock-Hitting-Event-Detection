# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 23:54:40 2023
@author: yuhsi
"""

import os
import csv
import random
import numpy as np
from pandas import *

# csv header
header = ['VideoName', 'ShotSeq', 'HitFrame', 'Hitter', 'RoundHead', 'Backhand', 'BallHeight',
              'LandingX', 'LandingY', 'HitterLocationX', 'HitterLocationY', 
              'DefenderLocationX', 'DefenderLocationY', 'BallType', 'Winner']


"""
data = read_csv('./event/00001_predict_denoise_event.csv')
eventList = data.event.tolist()
frameList = np.nonzero(eventList)[0]
print(eventList, frameList)
print(len(eventList))
print(np.sum(eventList))
"""



# csv data
data = []
eventPath = '/home/yuhsi/Badminton/src/TrackNetV2_pytorch/10-10Gray/event/'    ################################ 1. event-path ################################
eventFolder = os.listdir(eventPath)
eventFolder = sorted(eventFolder)
#print(eventFolder)

for f in eventFolder:

    ss = 1    # shotseq number

    csv_data = read_csv(eventPath + f)
    eventList = csv_data.event.tolist()
    frameList = np.nonzero(eventList)[0]

    vn = f.split('_')[0] + '.mp4'
    #print(eventList)
    #print(frameList)

    for i in range(len(frameList)):
        data.append([vn, ss, frameList[i], 'X', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'X'])
        ss+=1



with open('tracknetv2_pytorch_10-10Gray_denoise_eventDetection_X.csv', 'w', encoding='UTF8', newline='') as f:    ################################ 2. save-name ################################
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(data)
