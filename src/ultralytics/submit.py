import os
import cv2
import csv
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8x-pose-p6.pt')

dataPath = "/home/yuhsi/Badminton/src/yolov5/runs/detect/exp_draw/case1/"    ################################ 1. plotframe ################################
imgList = os.listdir(dataPath)
imgList = sorted(imgList)

csvFile = '/home/yuhsi/Badminton/src/ViT-pytorch_Winner/CodaLab_testdata_track1_hitter_mean_roundhead_mean_backhand_mean_ballheight_mean_LX_LY_case1_HD_balltype_vote_winner_mean_case1.csv'    ################################ 2. csv-file ################################
df = pd.read_csv(csvFile)
hitframe = df['HitFrame'].tolist()
landingx, landingy = df['LandingX'].tolist(), df['LandingY'].tolist()

ct = 0
for ii in imgList:
    ip = dataPath + ii
    img = cv2.imread(ip)
    res = model(img, imgsz=1280, conf=0.5, max_det=2)
    res_plotted = res[0].plot(boxes=False)    ################################ 3. pose-only ################################

    # 5 keypoints for the spine, 4 keypoints for the left arm, 4 keypoints for the right arm, 2 keypoints for the left leg, and 2 keypoints for the right leg.
    keypoint_data = res[0].keypoints.data.cpu().detach().numpy()
    #print(keypoint_data)
    try:
        x1l, y1l, x1r, y1r = round(keypoint_data[0][-1][0]), round(keypoint_data[0][-1][1]), round(keypoint_data[0][-2][0]), round(keypoint_data[0][-2][1])
        x2l, y2l, x2r, y2r = round(keypoint_data[1][-1][0]), round(keypoint_data[1][-1][1]), round(keypoint_data[1][-2][0]), round(keypoint_data[1][-2][1])
        
        # find the closer center coordinate of detected boxes
        cx1, cy1 = (x1l + x1r) // 2, (y1l + y1r) // 2
        cx2, cy2 = (x2l + x2r) // 2, (y2l + y2r) // 2
        ballx, bally = landingx[ct], landingy[ct]
        l1 = np.sqrt((cx1-int(ballx))**2 + (cy1-int(bally))**2)
        l2 = np.sqrt((cx2-int(ballx))**2 + (cy2-int(bally))**2)

        # find the hitter
        if l1 < l2:

            # find the closer node for the hitter
            d1 = np.sqrt((x1l-ballx)**2 + (y1l-bally)**2)
            d2 = np.sqrt((x1r-ballx)**2 + (y1r-bally)**2)
            if d1 < d2:
                df['HitterLocationX'][ct], df['HitterLocationY'][ct] = x1l, y1l
                res_plotted = cv2.rectangle(res_plotted, (x1l-5, y1l-5), (x1l+5, y1l+5), (0,0,0), -1)
                res_plotted = cv2.rectangle(res_plotted, (x1l-3, y1l-3), (x1l+3, y1l+3), (0,118,238), -1)
            else:
                df['HitterLocationX'][ct], df['HitterLocationY'][ct] = x1r, y1r
                res_plotted = cv2.rectangle(res_plotted, (x1r-5, y1r-5), (x1r+5, y1r+5), (0,0,0), -1)
                res_plotted = cv2.rectangle(res_plotted, (x1r-3, y1r-3), (x1r+3, y1r+3), (0,118,238), -1)

            # find the closer node for the defender
            d1 = np.sqrt((x2l-ballx)**2 + (y2l-bally)**2)
            d2 = np.sqrt((x2r-ballx)**2 + (y2r-bally)**2)
            if d1 < d2:
                df['DefenderLocationX'][ct], df['DefenderLocationY'][ct] = x2l, y2l
                res_plotted = cv2.rectangle(res_plotted, (x2l-5, y2l-5), (x2l+5, y2l+5), (0,0,0), -1)
                res_plotted = cv2.rectangle(res_plotted, (x2l-3, y2l-3), (x2l+3, y2l+3), (238,178,0), -1)
            else:
                res_plotted = cv2.rectangle(res_plotted, (x2r-5, y2r-5), (x2r+5, y2r+5), (0,0,0), -1)
                res_plotted = cv2.rectangle(res_plotted, (x2r-3, y2r-3), (x2r+3, y2r+3), (238,178,0), -1)

        else:
            d1 = np.sqrt((x2l-ballx)**2 + (y2l-bally)**2)
            d2 = np.sqrt((x2r-ballx)**2 + (y2r-bally)**2)
            if d1 < d2:
                df['HitterLocationX'][ct], df['HitterLocationY'][ct] = x2l, y2l
                res_plotted = cv2.rectangle(res_plotted, (x2l-5, y2l-5), (x2l+5, y2l+5), (0,0,0), -1)
                res_plotted = cv2.rectangle(res_plotted, (x2l-3, y2l-3), (x2l+3, y2l+3), (0,118,238), -1)
            else:
                df['HitterLocationX'][ct], df['HitterLocationY'][ct] = x2r, y2r
                res_plotted = cv2.rectangle(res_plotted, (x2r-5, y2r-5), (x2r+5, y2r+5), (0,0,0), -1)
                res_plotted = cv2.rectangle(res_plotted, (x2r-3, y2r-3), (x2r+3, y2r+3), (0,118,238), -1)
            d1 = np.sqrt((x1l-ballx)**2 + (y1l-bally)**2)
            d2 = np.sqrt((x1r-ballx)**2 + (y1r-bally)**2)
            if d1 < d2:
                df['DefenderLocationX'][ct], df['DefenderLocationY'][ct] = x1l, y1l
                res_plotted = cv2.rectangle(res_plotted, (x1l-5, y1l-5), (x1l+5, y1l+5), (0,0,0), -1)
                res_plotted = cv2.rectangle(res_plotted, (x1l-3, y1l-3), (x1l+3, y1l+3), (238,178,0), -1)
            else:
                df['DefenderLocationX'][ct], df['DefenderLocationY'][ct] = x1r, y1r
                res_plotted = cv2.rectangle(res_plotted, (x1r-5, y1r-5), (x1r+5, y1r+5), (0,0,0), -1)
                res_plotted = cv2.rectangle(res_plotted, (x1r-3, y1r-3), (x1r+3, y1r+3), (238,178,0), -1)
    except:
        pass

    cv2.imwrite('./pose_estimation/'+ii, res_plotted)
    ct += 1

df.to_csv('CodaLab_testdata_track1_hitter_mean_roundhead_mean_backhand_mean_ballheight_mean_LX_LY_case1_HD_balltype_vote_winner_mean_case1_v8pose.csv', index=False)
