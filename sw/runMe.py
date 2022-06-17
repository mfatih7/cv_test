import numpy as np
import os
import cv2 as cv
from matplotlib import pyplot as plt

from plotFunctions import plotFunctions as PF


parentDir=os.path.split(os.getcwd())[0]
inputDir=os.path.join(parentDir, 'inputs')
outputDir=os.path.join(parentDir, 'outputs')

if(os.path.exists(outputDir)==0):
    os.mkdir(outputDir)

SX = 1920
SY = 1080

CX = SX/2
CY = SY/2
INITIAL_FOCAL_LENGTH_GUESS = 100
N_IMAGES = 3
TRANSLATION_SCALE = 10000

# %% STEP 1 Finding Intrinsic Parameters Using Camera calibration Procedure with konwn 2D-3D correspondences

objectPoints = np.load(inputDir + '\\' + 'vr3d.npy')
imagePoints = np.load(inputDir + '\\' + 'vr2d.npy')

imageSize = (SX, SY)

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
initialCameraMatrix[0,0] = INITIAL_FOCAL_LENGTH_GUESS
initialCameraMatrix[1,1] = INITIAL_FOCAL_LENGTH_GUESS
initialCameraMatrix[2,2] = 1
initialCameraMatrix[0,2] = CX
initialCameraMatrix[1,2] = CY    

# Since calibration points are not planar initial intrinsic Matrix needed
# ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera([objectPoints], [imagePoints], imageSize , None, None)

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera([objectPoints], [imagePoints], imageSize, initialCameraMatrix, None, flags=flags)


# %% STEP 2 Finding SIFT Features In Images

sift                    = cv.SIFT_create()
imgList                 = []
imgWithKeypointsList    = []
kpList                  = []
desList                 = []

for i in range(N_IMAGES) :
    imNo = i+1
    imgWithKeypoints =[]
    
    img = cv.imread(inputDir + '\\' + 'img' + str(imNo) + '.png')
    imgList.append(img)
    
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)    
    kp, des = sift.detectAndCompute(gray,None)    
    
    imgWithKeypoints = cv.drawKeypoints(img,\
                                        kp, \
                                        outImage=np.array([]),\
                                        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)    
    
    kpList.append(kp)
    desList.append(des)
    imgWithKeypointsList.append(imgWithKeypoints)
    
    print('Number of SIFT features found in image ' + str(imNo) + ' is ' + str(len(desList[i])) )     

    cv.imwrite(outputDir + '\\' + 'SIFTkeypoints' + str(imNo) + '.png', imgWithKeypoints)  

# %% STEP 3 Applying Closeness Constraint to SIFT Features

bf = cv.BFMatcher()
src_ptsList = []
dst_ptsList = []

for i in range(N_IMAGES-1) :
    
    matches = bf.knnMatch(desList[0], desList[i+1], k=2)
    
    print('Number of matches between image 1 and image ' + str(i+2) + ' is ' + str(len(matches)) )
    
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:            
            good.append(m)    
    
    src_pts = np.float32([ kpList[0][m.queryIdx].pt for m in good ])
    dst_pts = np.float32([ kpList[i+1][m.trainIdx].pt for m in good ])
    
    print('Number of matches after closeness elimination is ' + str(len(src_pts)) )  
    
    src_ptsList.append(src_pts)
    dst_ptsList.append(dst_pts)

# %% STEP 4 Finding Essential Matrices (E)

Elist = []
maskList = []

for i in range(N_IMAGES-1) :
    
    src_pts = src_ptsList[i]
    dst_pts = dst_ptsList[i]
    
    E, mask = cv.findEssentialMat(src_pts,\
                                  dst_pts,\
                                  cameraMatrix,\
                                  method=cv.RANSAC,\
                                  threshold=1,\
                                  maxIters=1000    )

    Elist.append(E)
    maskList.append(mask)

# %% STEP 5 Checking Correspondences on Images

for i in range(N_IMAGES-1) :
    
    stackedImage = np.hstack( (imgList[0], imgList[i+1]) )
    
    src_ptsD_draw =np.int16(src_ptsList[i])
    dst_ptsD_draw =np.int16(dst_ptsList[i])
    maskDraw = maskList[i]
    
    matchCounter = 0
    for j in range(len(mask)): 
        if(maskDraw[j,0]==1):
            matchCounter = matchCounter +1
            if(matchCounter%50==0):
                color1 = (list(np.random.choice(range(256), size=3)))  
                color =[int(color1[0]), int(color1[1]), int(color1[2])] 
                cv.line(stackedImage, (src_ptsD_draw[j,0], src_ptsD_draw[j,1]), (SX+dst_ptsD_draw[j,0], dst_ptsD_draw[j,1]), color, 2)

    cv.imwrite(outputDir + '\\' + 'CorrespondenceCheckOnImage' + str(1) + 'andImage' + str(i+2) + '.png', stackedImage)

# %% STEP 6 Decomposing Essential Matrix (E) into Rotation Matrix (R) and Translation Vector (t)

Rlist = []
tList = []

for i in range(N_IMAGES-1) :
    
    correctCheckCount = []
    R = []
    t = []
    
    E = Elist[i]
    src_pts = src_ptsList[i]
    dst_pts = dst_ptsList[i]
    mask = maskList[i]
    
#   Not using decomposeEssentialMat() since it gives 2 results for rotation and translation   
    # R1, R2, t = cv.decomposeEssentialMat(E)

#   Using recoverPose() since it also controls for rotation matrices and translation matrices with Cheirality Condition 
    correctCheckCount, R, t, _ = cv.recoverPose(E, src_pts, dst_pts, cameraMatrix, mask)
    
    Rlist.append(R)
    tList.append(t)
    
    print(str(int(100*correctCheckCount/len(mask))) + ' % of the matches remain correct after cheirality check in recoverPose() for image 1 - image ' + str(i+2) + ' pair')  


# %% STEP 7 PLOTTING OUTPUT FIGURES

tMatrix = np.zeros((3,3))
for i in range(N_IMAGES-1) :
    # Since direction of t is from image 2 and image 3 to image 1
    # t must be multiplied with scalar -1 to reverse the direction of the vector
    tMatrix[:,i+1]=np.multiply(np.squeeze(tList[i]), -1) 

PF.plotCameraTrajectory(outputDir, tMatrix, TRANSLATION_SCALE)
PF.plotCameraRotation(outputDir, SX, SY, Rlist, tMatrix, TRANSLATION_SCALE)
         