import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt


class plotFunctions:

    def plotCameraTrajectory(outputDir, tMatrix, TRANSLATION_SCALE):
    
        N_POINTS_ON_TRAJECTORIES = 100
        ax = plt.axes(projection="3d")
        
        tMatrix = np.multiply(tMatrix, TRANSLATION_SCALE)
        
        xMax = np.max(tMatrix[0,:])
        yMax = np.max(tMatrix[1,:])
        zMax = np.max(tMatrix[2,:])

        xMin = np.min(tMatrix[0,:])
        yMin = np.min(tMatrix[1,:])
        zMin = np.min(tMatrix[2,:])
        
        xDiff = xMax-xMin
        yDiff = yMax-yMin
        zDiff = zMax-zMin       
        
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        
        ax.set_xlim(xMin-xDiff/10, xMax+xDiff/10 )
        ax.set_ylim(zMin-zDiff/10, zMax+zDiff/10 )
        ax.set_zlim(yMin-yDiff/10, yMax+yDiff/10 )        

        for i in range(np.size(tMatrix,1)):
            
            ax.scatter(tMatrix[0][i], tMatrix[2][i], tMatrix[1][i])
            
            if(i>0):            
                x = np.squeeze(np.linspace(tMatrix[0][0], tMatrix[0][i], N_POINTS_ON_TRAJECTORIES))
                y = np.squeeze(np.linspace(tMatrix[1][0], tMatrix[1][i], N_POINTS_ON_TRAJECTORIES))
                z = np.squeeze(np.linspace(tMatrix[2][0], tMatrix[2][i], N_POINTS_ON_TRAJECTORIES))
                   
                ax.plot3D(x,z,y)
        
        ax.legend(['Camera Translation from Image 1 to Image 2', 'Camera Translation from Image 1 to Image 3'])
        

        ax.view_init(elev=36, azim=-51)
        plt.savefig(outputDir + '\\' + 'Camera Translations_Fig1' )
        
        ax.view_init(elev=60, azim=-5)
        plt.savefig(outputDir + '\\' + 'Camera Translations_Fig2' )
    
    def plotCameraRotation():
    
        print(1)