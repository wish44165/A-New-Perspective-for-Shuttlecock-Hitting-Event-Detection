
import os
import cv2
import numpy as np
import time

def imgradient(img):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0) # Find x and y gradients
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1)

    # Find magnitude and angle
    magnitude = np.sqrt(sobelx**2.0 + sobely**2.0)
    #angle = np.arctan2(sobely, sobelx) * (180 / np.pi)
    
    #return magnitude, angle
    return magnitude


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


    # 1st frame
    _, frameRGB0 = cap.read()
    frameGray0 = cv2.cvtColor(frameRGB0, cv2.COLOR_BGR2GRAY)
    mgrad0 = imgradient(frameGray0)
    # 2nd frame
    _, frameRGB1 = cap.read()
    frameGray1 = cv2.cvtColor(frameRGB1, cv2.COLOR_BGR2GRAY)
    mgrad1 = imgradient(frameGray1)

    height, width, _ = frameRGB0.shape
    frameSize = (width, height)

    FPS = cap.get(cv2.CAP_PROP_FPS)
    print('fps =', FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(savePath + vn[:-4] + '_xgg.mp4', fourcc, FPS, frameSize)    ################################ 3. name ################################

    # Add two frames to maintain the same frame number
    ## frame 0
    nullc1 = (frameRGB0[...,2])[:, :, np.newaxis]
    nullc2 = np.uint8(np.zeros(nullc1.shape))
    nullFrame0 = np.concatenate((nullc2, nullc2, nullc2), axis=2)    ################################ 4. type ################################
    out.write(nullFrame0)
    #cv2.imshow('frame', nullFrame0)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # display
    cv2.imshow('frame', nullFrame0)
    if cv2.waitKey(1) == ord('q'):
        break

    ## frame 1
    delta_frame = 0.5 * np.abs(mgrad1 - mgrad0)
    df_local_mean = cv2.boxFilter(delta_frame, -1, (3,3), normalize=True)
    ndf_local_mean = (df_local_mean - np.min(df_local_mean)) / (np.max(df_local_mean) - np.min(df_local_mean))
    c2 = np.uint8(np.round(255*ndf_local_mean))[:, :, np.newaxis]
    c3 = (frameRGB1[...,2])[:, :, np.newaxis]
    ndfrgb = np.concatenate((nullc2, c2, c2), axis=2)    ################################ 5. type ################################
    out.write(ndfrgb)
    #cv2.imshow('frame', ndfrgb)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #break

    # display
    cv2.imshow('frame', ndfrgb)
    if cv2.waitKey(1) == ord('q'):
        break

    while True:
        ret, frameRGB2 = cap.read()
        if not ret:
            break
        frameGray2 = cv2.cvtColor(frameRGB2, cv2.COLOR_BGR2GRAY)

        mgrad2 = imgradient(frameGray2)
        delta_frame = 0.5 * np.abs(mgrad2 - mgrad0)
        df_local_mean = cv2.boxFilter(delta_frame, -1, (3,3), normalize=True)
        ndf_local_mean = (df_local_mean - np.min(df_local_mean)) / (np.max(df_local_mean) - np.min(df_local_mean))

        c2 = np.uint8(np.round(255*ndf_local_mean))[:, :, np.newaxis]
        c3 = (frameRGB1[...,2])[:, :, np.newaxis]
        
        #ndfrgb = np.concatenate((c3, c2, c2), axis=2)    ################################ 6. type ################################
        ndfrgb = np.concatenate((np.uint8(np.zeros(c2.shape)), c2, c2), axis=2)

        # display
        cv2.imshow('frame', ndfrgb)
        if cv2.waitKey(1) == ord('q'):
            break


        # save RT video
        out.write(ndfrgb)
        

        # update
        mgrad0 = mgrad1
        mgrad1 = mgrad2
        frameGray0 = frameGray1
        frameGray1 = frameGray2
        frameRGB1 = frameRGB2


    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print('output fps=', videoLength / (time.time() - startTime))


    cap = cv2.VideoCapture(savePath + vn[:-4] + '_xgg.mp4')    ################################ 7. check ################################
    new_videoLength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('new_videoLength =', new_videoLength)
    cap.release()
    
    #break

    if new_videoLength != videoLength:
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        break

