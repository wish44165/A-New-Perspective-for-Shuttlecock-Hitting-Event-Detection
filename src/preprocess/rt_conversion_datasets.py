
import os
import cv2
import numpy as np
import time

def rt_conversion():
    pass

valPath = '/home/yuhsi/Badminton/data/part1/val/'    ################################ 1. path ################################
valFolders = os.listdir(valPath)
valFolders = sorted(valFolders)
#print(valFolders)

savePath = '/home/yuhsi/Badminton/src/preprocess/val_test_xgg/'    ################################ 2. path ################################

for vf in valFolders:

    startTime = time.time()

    vp = valPath + vf + '/' + vf + '.mp4'

    print('================', vp, '================')

    vn = vp.split('/')[-1]

    cap = cv2.VideoCapture(vp)
    videoLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('videoLength =', videoLength)

    _, frameRGB0 = cap.read()
    height, width, _ = frameRGB0.shape
    frameSize = (width, height)

    FPS = cap.get(cv2.CAP_PROP_FPS)
    print('fps =', FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(savePath + vn[:-4] + '_xgg.mp4', fourcc, FPS, frameSize)    ################################ 3. name ################################

    # Add two frames to maintain the same frame number

    ## frame 1

    # display
    cv2.imshow('frame', ndfrgb)
    if cv2.waitKey(1) == ord('q'):
        break

    while True:
        ret, frameRGB2 = cap.read()

        if not ret:
            break

        ndfrgb = rt_conversion()

        # display
        cv2.imshow('frame', ndfrgb)
        if cv2.waitKey(1) == ord('q'):
            break

        # save RT video
        out.write(ndfrgb)

        # update

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print('output fps=', videoLength / (time.time() - startTime))