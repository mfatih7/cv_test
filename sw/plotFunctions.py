import numpy as np
import os
import cv2 as cv
import matplotlib.pyplot as plt


class plotFunctions:

    def plotCameraTrajectory(outputDir, t, imNo, TRANSLATION_SCALE):
    
        print(2)
        
        ax = plt.axes(projection="3d")

        x = np.squeeze(np.linspace(0, t[0], 100))
        y = np.squeeze(np.linspace(0, t[1], 100))
        z = np.squeeze(np.linspace(0, t[2], 100))
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        ax.set_xlim(-10, 1000 )
        ax.set_ylim(-10, 10 )
        ax.set_zlim(-10, 1000 )
        
        ax.plot3D(x,y,z)

        plt.show()

        plt.savefig('filename.png')
    
    def plotCameraRotation():
    
        print(1)