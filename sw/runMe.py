import numpy as np
import os
import cv2 as cv
from matplotlib import pyplot as plt

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
# ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera([objectPoints], [imagePoints], imageSize , None, None)

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera([objectPoints], [imagePoints], imageSize, initialCameraMatrix, None, flags=flags)

imgList                 = []
imgWithKeypointsList    = []
kpList                  = []
desList                 = []
sift                    = cv.SIFT_create()

for i in range(N_IMAGES) :
    imNo = i+1
    imgWithKeypoints =[]
    
    img = cv.imread(inputDir + '\\' + 'img' + str(imNo) + '.png')
    imgList.append(img)

    plt.imshow(imgList[i]),plt.show()
    
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)    
    kp, des = sift.detectAndCompute(gray,None)
    
    
    imgWithKeypoints = cv.drawKeypoints(img,\
                                        kp, \
                                        outImage=np.array([]),\
                                        flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imwrite(outputDir + '\\' + 'sift_keypoints' + str(imNo) + '.png', imgWithKeypoints)
    
    kpList.append(kp)
    desList.append(des)
    imgWithKeypointsList.append(imgWithKeypoints)

    plt.imshow(imgWithKeypointsList[i]),plt.show()    





bf = cv.BFMatcher()

for i in range(N_IMAGES-1) :
    print(i+1)
    
    matches = bf.knnMatch(desList[0], desList[i+1], k=2)
    
    print(len(desList[0]))        
    print(len(desList[i+1])) 
    
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:            
            good.append(m)
            
    src_pts = np.float32([ kpList[0][m.queryIdx].pt for m in good ])
    dst_pts = np.float32([ kpList[i+1][m.trainIdx].pt for m in good ])
    
    print(len(src_pts))        
    print(len(dst_pts))    
    
    E, mask = cv.findEssentialMat(src_pts,\
                                  dst_pts,\
                                  cameraMatrix,\
                                  method=cv.RANSAC,\
                                  threshold=1,\
                                  maxIters=1000    )
    print(len(mask))
    
    A = np.hstack( (imgList[0], imgList[i+1]) )
    
    src_ptsD_draw =np.int16(src_pts)
    dst_ptsD_draw =np.int16(dst_pts)
    
    matchCounter = 0
    for j in range(len(mask)):        
    # for j in range(1):  
        if(mask[j,0]==1):
            matchCounter = matchCounter +1
            if(matchCounter%100==0):
                color1 = (list(np.random.choice(range(256), size=3)))  
                color =[int(color1[0]), int(color1[1]), int(color1[2])] 
                # cv.line(A, (20,10), (100,10), (255,0,0), 1)
                cv.line(A, (src_ptsD_draw[j,0], src_ptsD_draw[j,1]), (CX*2+dst_ptsD_draw[j,0], dst_ptsD_draw[j,1]), color, 1)

    
    plt.imshow(A),plt.show()
    cv.imwrite(outputDir + '\\' + 'A' + str(i) + '.png', A)    
        
    # for j in range(len(mask)):
        
    # img2 = cv.drawMatches(imgList[0], src_pts, imgList[i+1], dst_pts, mask, imgList[0] )   
    # cv.imwrite(outputDir + '\\' + 'asd' + img2 + '.png', img)
        
    R1, R2, t = cv.decomposeEssentialMat(E)
        
    dst1, jacobian1 = cv.Rodrigues(R1)
    dst2, jacobian2 = cv.Rodrigues(R2)
    
    # M, mask = cv.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    # matchesMask = mask.ravel().tolist()        
            
            
            