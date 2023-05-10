import os
import csv
import numpy as np
import pandas as pd

csvPath1 = '/home/yuhsi/Badminton/src/ViT-pytorch_BallHeight/golfdb_G3_fold5_iter3000_val_test_hitter_vote_roundhead_vote_backhand_vote_ballheight_vote.csv'    ################################ 1. csvPath ################################
csvPath2 = '/home/yuhsi/Badminton/src/ViT-pytorch_BallHeight/golfdb_G3_fold5_iter3000_val_test_hitter_mean_roundhead_mean_backhand_mean_ballheight_mean.csv'    ################################ 1. csvPath ################################
filePath = '/home/yuhsi/Badminton/src/TrackNetV2_pytorch/10-10Gray/denoise/'

df1 = pd.read_csv(csvPath1)
df2 = pd.read_csv(csvPath2)
df_videoname = df1['VideoName']
df_hitframe = df1['HitFrame']

print(df_videoname)
print(df_hitframe)

allFiles = os.listdir(filePath)
allFiles = sorted(allFiles)
#print(allFiles)


LX = []
LY = []

for i in range(len(df_videoname)):
    fp = filePath + df_videoname[i].split('.')[0] + '_predict_denoise.csv'
    df_csv = pd.read_csv(fp)

    #### ???? ####
    if df_hitframe[i] < len(df_csv['X'].tolist()):
        x = round(df_csv['X'].tolist()[df_hitframe[i]])
    else:
        x = 0
    if df_hitframe[i] < len(df_csv['Y'].tolist()):
        y = round(df_csv['Y'].tolist()[df_hitframe[i]])
    else:
        y = 0

    LX.append(x)
    LY.append(y)

print(len(LX))
print(len(LY))

df1['LandingX'] = LX    ################################ 4. attribute ################################
df1['LandingY'] = LY
df2['LandingX'] = LX    ################################ 4. attribute ################################
df2['LandingY'] = LY
df1.to_csv('golfdb_G3_fold5_iter3000_val_test_hitter_vote_roundhead_vote_backhand_vote_ballheight_vote_LXY.csv', index=False)    ################################ 5. csv-name ################################
df2.to_csv('golfdb_G3_fold5_iter3000_val_test_hitter_mean_roundhead_mean_backhand_mean_ballheight_mean_LXY.csv', index=False)    ################################ 5. csv-name ################################



