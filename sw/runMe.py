import numpy as np
import os
import cv2 as cv

CX = 960
CY = 540
INTRINSIC_GUESS = 100
N_IMAGES = 3

imageSize = (CX*2, CY*2)

parentDir=os.path.split(os.getcwd())[0]
inputDir=os.path.join(parentDir, 'inputs')
outputDir=os.path.join(parentDir, 'outputs')

objectPoints = np.load(inputDir + '\\' + 'vr3d.npy')
imagePoints = np.load(inputDir + '\\' + 'vr2d.npy')


flags = cv.CALIB_USE_INTRINSIC_GUESS + \
        cv.CALIB_FIX_PRINCIPAL_POINT + \
        cv.CALIB_FIX_ASPECT_RATIO + \
        cv.CALIB_ZERO_TANGENT_DIST + \
        cv.CALIB_FIX_K1 + \
        cv.CALIB_FIX_K2 + \
        cv.CALIB_FIX_K3 + \
        cv.CALIB_FIX_K4 + \
        cv.CALIB_FIX_K5 + \
        cv.CALIB_FIX_K6 
        
initialCameraMatrix = np.zeros((3,3),'float32')
initialCameraMatrix[0,0] = INTRINSIC_GUESS
initialCameraMatrix[1,1] = INTRINSIC_GUESS
initialCameraMatrix[2,2] = 1
initialCameraMatrix[0,2] = CX
initialCameraMatrix[1,2] = CY    

# Since calibration points are not planar initial intrinsic Matrix needed
# ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera([objectPoints], [imagePoints], imageSize , None, None)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera([objectPoints], [imagePoints], imageSize, initialCameraMatrix, None, flags=flags)


sift = cv.SIFT_create()
kpList = []


for i in range(N_IMAGES) :
    imNo = i+1
    
    img = cv.imread(inputDir + '\\' + 'img' + str(imNo) + '.png')
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    kp = sift.detect(gray,None)    
    # img=cv.drawKeypoints(gray,kp,img)
    img=cv.drawKeypoints(gray,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite(outputDir + '\\' + 'sift_keypoints' + str(imNo) + '.png', img)
    
    kpList.append(kp)
    
