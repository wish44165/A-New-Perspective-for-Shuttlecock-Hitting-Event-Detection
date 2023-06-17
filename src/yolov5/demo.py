import os
import re
import csv
import cv2
import numpy as np
import pandas as pd

def drawline(img,pt1,pt2,color,thickness=1,style='dotted',gap=20):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)

    if style=='dotted':
        for p in pts:
            cv2.circle(img,p,thickness,color,-1)
    else:
        s=pts[0]
        e=pts[0]
        i=0
        for p in pts:
            s=e
            e=p
            if i%2==1:
                cv2.line(img,s,e,color,thickness)
            i+=1
    return img


WIDTH, HEIGHT = 1280, 720
CASE = 1    ################################ 0. case ################################

csvPath = '/home/yuhsi/Badminton/src/TrackNetV2_pytorch/golfdb_G3_fold5_iter3000_val_test_hitter_mean_roundhead_mean_backhand_mean_ballheight_mean_LXY.csv'    ################################ 1. csvPath ################################
filePath = '/home/yuhsi/Badminton/src/yolov5/runs/detect/exp/labels/'    ################################ 2. yolo detected path ################################
drawPath = '/home/yuhsi/Badminton/src/yolov5/runs/detect/exp/'    ################################ 2. yolo detected path ################################
detectedPath = '/home/yuhsi/Badminton/src/postprocess/HitFrame_yolo/'    ################################ 3. hitframe for yolo path ################################

df = pd.read_csv(csvPath)
df_videoname = df['VideoName']
df_hitframe = df['HitFrame']

#print(df_videoname)
#print(df_hitframe)

allFiles = os.listdir(filePath)
allFiles = sorted(allFiles)
#print(allFiles)


allImg = os.listdir(detectedPath)
allImg = sorted(allImg)
#print(allImg)

yoloImg = os.listdir(drawPath)
yoloImg.remove('labels')
yoloImg = sorted(yoloImg)
#print('yoloImg[-1] =', yoloImg[-1])    # yoloImg[-1] = labels
#print(yoloImg)
#print('len(yoloImg) =', len(yoloImg))







# classes: 'A':0, 'B':1, 'ball':2 
personList = [{'A':[0,0,0,0],'B':[0,0,0,0]} for i in range(len(allImg))]    # x1, y1, x2, y2
img_ct = 0
ct = 0
for i in range(len(df_videoname)):
    s1 = df_videoname[i].split('.')[0]
    if len(str(df_hitframe[i])) == 1:
        s2 = '0000' + str(df_hitframe[i])
    elif len(str(df_hitframe[i])) == 2:
        s2 = '000' + str(df_hitframe[i])
    elif len(str(df_hitframe[i])) == 3:
        s2 = '00' + str(df_hitframe[i])
    elif len(str(df_hitframe[i])) == 4:
        s2 = '0' + str(df_hitframe[i])
    else:
        s2 = '' + str(df_hitframe[i])
    #print(s1, s2)
    tp = filePath + s1 + '_' + s2 + '.txt'
    try:
        with open(tp) as f:
            lines = f.readlines()
        for j in range(len(lines)):
            l = lines[j]
            cj = l[0]
            bboxj = [float(a) for a in re.findall("\d+\.\d+", l)]    # ratio of cx, cy, w, h
            pixbboxj = [round(WIDTH*bboxj[0]), round(HEIGHT*bboxj[1]), round(WIDTH*bboxj[2]), round(HEIGHT*bboxj[3])]    # pixel value of cx, cy, w, h
            lrbbox = [round(WIDTH*bboxj[0]) - round(WIDTH*bboxj[2])//2, round(HEIGHT*bboxj[1]) - round(HEIGHT*bboxj[3])//2, round(WIDTH*bboxj[0]) + round(WIDTH*bboxj[2])//2, round(HEIGHT*bboxj[1]) + round(HEIGHT*bboxj[3])//2]    # x1, y1, x2, y2
            area = round(WIDTH*bboxj[2]) * round(HEIGHT*bboxj[3])
            #print(cj, bboxj, pixbboxj, lrbbox, area)
            if cj == str(0):    # 'A'
                area_o = float(personList[img_ct]['A'][-2]) * float(personList[img_ct]['A'][-1])
                if area > area_o:
                    personList[img_ct]['A'] = lrbbox
            elif cj == str(1):    # 'B'
                area_o = float(personList[img_ct]['B'][-2]) * float(personList[img_ct]['B'][-1])
                if area > area_o:
                    personList[img_ct]['B'] = lrbbox
    except:
        print('tp =', tp)
    if len(lines) < 2:
        ct+=1

    img_ct+=1

print('number of length of lines less than 2:', ct)
print('length of person list:', len(personList))




df_hitter = df['Hitter']
df_landingx = df['LandingX'].tolist()
df_landingy = df['LandingY'].tolist()
#print(df_landingx)
#print(df_landingy)


# Radius of circle
radius = 5
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 2

newY = []

hitterlocationX = []
hitterlocationY = []
defenderlocationX = []
defenderlocationY = []




cap = cv2.VideoCapture('/home/yuhsi/Badminton/src/yolov5/runs/detect/exp_demo/00399.mp4')
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
print(length, width, height, fps)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('00399.mp4',fourcc, fps, (width, height),True)

drawList = df_hitframe.tolist()[-21:]
print(drawList)
personList = personList[-21:]
print(personList)
df_landingx = df_landingx[-21:]
df_landingy = df_landingy[-21:]
i = 0

for fni in range(length):
    ret, im = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if fni in set(drawList):

        A_cx, A_cy, B_cx, B_cy = (personList[i]['A'][0] + personList[i]['A'][2]) // 2, (personList[i]['A'][1] + personList[i]['A'][3]) // 2, (personList[i]['B'][0] + personList[i]['B'][2]) // 2, (personList[i]['B'][1] + personList[i]['B'][3]) // 2
        print('personList[i] =', personList[i])
        print('A_cx, A_cy, B_cx, B_cy =', A_cx, A_cy, B_cx, B_cy)
        ballx, bally = df_landingx[i], df_landingy[i]
        print('ballx, bally =', ballx, bally)
        #im = cv2.imread(drawPath + yoloImg[i])


        if df_hitter[i] == 'A':
            #print(personList[i]['A'])    # x1, y1, x2, y2
            h_x1, h_x2, h_y2 = personList[i]['A'][0], personList[i]['A'][2], personList[i]['A'][3]
            d_x1, d_x2, d_y2 = personList[i]['B'][0], personList[i]['B'][2], personList[i]['B'][3]
        else:
            #print(personList[i]['B'])    # x1, y1, x2, y2
            h_x1, h_x2, h_y2 = personList[i]['B'][0], personList[i]['B'][2], personList[i]['B'][3]
            d_x1, d_x2, d_y2 = personList[i]['A'][0], personList[i]['A'][2], personList[i]['A'][3]

        hitter_dist_x1 = np.abs(ballx - h_x1)
        hitter_dist_x2 = np.abs(ballx - h_x2)
        if hitter_dist_x1 < hitter_dist_x2:
            hitter_x, hitter_y = h_x1, h_y2
        else:
            hitter_x, hitter_y = h_x2, h_y2
        defender_dist_x1 = np.abs(ballx - d_x1)
        defender_dist_x2 = np.abs(ballx - d_x2)
        if defender_dist_x1 < defender_dist_x2:
            defender_x, defender_y = d_x1, d_y2
        else:
            defender_x, defender_y = d_x2, d_y2

        #print('hitter_x, hitter_y, defender_x, defender_y =', hitter_x, hitter_y, defender_x, defender_y)
        hitterlocationX.append(hitter_x)
        hitterlocationY.append(hitter_y)
        defenderlocationX.append(defender_x)
        defenderlocationY.append(defender_y)

        im = cv2.rectangle(im, (hitter_x-5, hitter_y-5), (hitter_x+5, hitter_y+5), (0,118,238), -1)
        im = cv2.rectangle(im, (defender_x-5, defender_y-5), (defender_x+5, defender_y+5), (238,178,0),-1)


        if CASE == 1:     # case 1: depending on hitter vit
            if A_cx + A_cy + B_cx + B_cy == 0:    # no people detected
                newY.append(0)
            else:
                if ballx == 0:
                    newY.append(0)
                else:
                    if df_hitter[i] == 'A':
                        newY.append(personList[i]['A'][3])
                        # draw
                        center_coordinates = (ballx, personList[i]['A'][3])
                        im = cv2.line(im, (ballx, bally), (A_cx, A_cy), (139,0,0), thickness)
                        im = cv2.line(im, (ballx, bally), (B_cx, B_cy), (237,149,100), thickness)
                    elif df_hitter[i] == 'B':
                        newY.append(personList[i]['B'][3])
                        # draw
                        center_coordinates = (ballx, personList[i]['B'][3])
                        im = cv2.line(im, (ballx, bally), (B_cx, B_cy), (139,0,0), thickness)
                        im = cv2.line(im, (ballx, bally), (A_cx, A_cy), (237,149,100), thickness)

        else:    # case 2: depending on distance between ball and center of bbox
            if A_cx + A_cy + B_cx + B_cy == 0:    # no people detected
                newY.append(0)
            else:
                if ballx == 0:
                    newY.append(0)
                else:
                    # distance
                    distA = np.sqrt((ballx - A_cx)**2 + (bally - A_cy)**2)
                    distB = np.sqrt((ballx - B_cx)**2 + (bally - B_cy)**2)
                    # x-distance
                    distA_x = np.abs(ballx - A_cx)
                    distB_x = np.abs(ballx - B_cx)
                    #print('distA, distB =', distA, distB)
                    if distA + distA_x <= distB + distB_x:
                        newY.append(personList[i]['A'][3])
                        # draw
                        center_coordinates = (ballx, personList[i]['A'][3])
                        im = cv2.line(im, (ballx, bally), (A_cx, A_cy), (139,0,0), thickness)
                        im = cv2.line(im, (ballx, bally), (B_cx, B_cy), (237,149,100), thickness)
                    elif distB + distB_x < distA + distA_x:
                        newY.append(personList[i]['B'][3])
                        # draw
                        center_coordinates = (ballx, personList[i]['B'][3])
                        im = cv2.line(im, (ballx, bally), (B_cx, B_cy), (139,0,0), thickness)
                        im = cv2.line(im, (ballx, bally), (A_cx, A_cy), (237,149,100), thickness)

        im = cv2.circle(im, (ballx, bally), radius, (0, 255, 255), -1)    # ball
        if ballx != 0:
            #im = cv2.circle(im, center_coordinates, radius, (0,0,0), -1)    # projection of ball
            im = cv2.line(im, (center_coordinates[0]-3, center_coordinates[1]-3), (center_coordinates[0]+3, center_coordinates[1]+3), (0,0,0), thickness)
            im = cv2.line(im, (center_coordinates[0]-3, center_coordinates[1]+3), (center_coordinates[0]+3, center_coordinates[1]-3), (0,0,0), thickness)
            # https://stackoverflow.com/questions/26690932/opencv-rectangle-with-dotted-or-dashed-lines
            im = drawline(im, (ballx, bally), center_coordinates, (0,0,0), thickness, style='dotted', gap=10)
            #print(center_coordinates)
        
        i+=1

    # display
    cv2.imshow('frame', im)
    out.write(im)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()