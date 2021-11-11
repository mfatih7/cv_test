import numpy as np
import os
import cv2

CX = 960
CY = 540
INTRINSIC_GUESS = 100

parentDir=os.path.split(os.getcwd())[0]
inputDir=os.path.join(parentDir, 'inputs')
outputDir=os.path.join(parentDir, 'outputs')

objpoints = np.load(inputDir + '\\' + 'vr3d.npy')
imgpoints = np.load(inputDir + '\\' + 'vr2d.npy')


flags = cv2.CALIB_USE_INTRINSIC_GUESS + \
        cv2.CALIB_FIX_PRINCIPAL_POINT + \
        cv2.CALIB_FIX_ASPECT_RATIO + \
        cv2.CALIB_ZERO_TANGENT_DIST + \
        cv2.CALIB_FIX_K1 + \
        cv2.CALIB_FIX_K2 + \
        cv2.CALIB_FIX_K3 + \
        cv2.CALIB_FIX_K4 + \
        cv2.CALIB_FIX_K5 + \
        cv2.CALIB_FIX_K6 
        
initialCameraMatrix = np.zeros((3,3),'float32')
initialCameraMatrix[0,0] = INTRINSIC_GUESS
initialCameraMatrix[1,1] = INTRINSIC_GUESS
initialCameraMatrix[2,2] = 1
initialCameraMatrix[0,2] = CX
initialCameraMatrix[1,2] = CY    

# Since calibration points are not planar initial intrinsic Matrix needed
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objpoints], [imgpoints], (CX*2, CY*2), None, None)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objpoints], [imgpoints], (CX*2, CY*2), initialCameraMatrix, None, flags=flags)





# C = np.tile(np.array([CX, CY]), (np.size(A,0),1))

# A = np.subtract(A,C)


# img1 = cv2.imread(inputDir + '\\' + 'img1.png')
# dimensions1 = img1.shape

# img2 = cv2.imread(inputDir + '\\' + 'img1.png')
# dimensions2 = img2.shape

# img3 = cv2.imread(inputDir + '\\' + 'img1.png')
# dimensions3 = img3.shape

# dimensions1[0]

# # A1 = np.subtract(A, np.repeat())
