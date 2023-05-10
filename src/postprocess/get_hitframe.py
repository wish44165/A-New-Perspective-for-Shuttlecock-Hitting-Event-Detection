
import os
import cv2
import numpy as np
import pandas as pd

csvPath = '/home/yuhsi/Badminton/src/postprocess/golfdb_G3_fold5_iter3000_val_test_X.csv'    ################################ 1. path ################################
savePath = '/home/yuhsi/Badminton/src/postprocess/HitFrame/1/'    ################################ 2. path ################################
valPath = '/home/yuhsi/Badminton/data/part1/val/'    ################################ 3. path ################################
data = pd.read_csv(csvPath)

vns = [a for a in data['VideoName']]
hits = [a for a in data['HitFrame']]

print(len(vns), len(hits))

for i in range(len(vns)):
    videoPath = valPath + vns[i].split('.')[0] + '/' + vns[i]

    cap = cv2.VideoCapture(videoPath)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for k in range(frame_count):
        ret, frame = cap.read()
        frame = frame[:, 280:1280-280]    ################################ 4. crop ################################
        if ret:
            if k == hits[i]:
                if len(str(k)) == 5:
                    cv2.imwrite(savePath + vns[i].split('.')[0] + '_' + str(k) + '.jpg', frame)
                if len(str(k)) == 4:
                    cv2.imwrite(savePath + vns[i].split('.')[0] + '_0' + str(k) + '.jpg', frame)
                if len(str(k)) == 3:
                    cv2.imwrite(savePath + vns[i].split('.')[0] + '_00' + str(k) + '.jpg', frame)
                if len(str(k)) == 2:
                    cv2.imwrite(savePath + vns[i].split('.')[0] + '_000' + str(k) + '.jpg', frame)
                if len(str(k)) == 1:
                    cv2.imwrite(savePath + vns[i].split('.')[0] + '_0000' + str(k) + '.jpg', frame)
                break
    cap.release()
    cv2.destroyAllWindows()

# check
print(len(os.listdir(savePath)))
