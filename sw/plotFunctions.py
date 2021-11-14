import numpy as np
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
                   
            if(i==1):
                ax.plot3D(x,z,y,color='red')
            elif(i==2):
                ax.plot3D(x,z,y,color='green')
        
        ax.legend(['Camera Translation from Image 1 to Image 2', 'Camera Translation from Image 1 to Image 3'])        

        ax.view_init(elev=36, azim=-51)
        plt.savefig(outputDir + '\\' + 'Camera Translations_Fig1' )
        
        ax.view_init(elev=60, azim=-5)
        plt.savefig(outputDir + '\\' + 'Camera Translations_Fig2' )
        plt.close()
    
    def plotCameraRotation(outputDir, SX, SY, Rlist, tMatrix, TRANSLATION_SCALE):
        
        N_POINTS_ON_EDGES = 100
        CAMERA_SCALE =1
        
        sx=SX/CAMERA_SCALE
        sy=SY/CAMERA_SCALE
    
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
        
        pDiff = max(xDiff, yDiff, zDiff)
        pMax = max(xMax, yMax, zMax)
        pMin = max(xMin, yMin, zMin)
        
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        
        ax.set_xlim(pMin-pDiff/10, pMax+pDiff/10 )
        ax.set_ylim(pMin-pDiff/10, pMax+pDiff/10 )
        ax.set_zlim(pMin-pDiff/10, pMax+pDiff/10 )    
        
        
        for i in range(np.size(tMatrix,1)):
            
            if i>0:
                R = Rlist[i-1]   
            
            for j in range(4):
                
                edges =np.zeros( (N_POINTS_ON_EDGES,3) )
                
                if j==0:
                    edges[:,0] = np.squeeze(np.linspace(tMatrix[0,i]-sx/2, tMatrix[0,i]+sx/2, N_POINTS_ON_EDGES ))
                    edges[:,1] = np.squeeze(np.linspace(tMatrix[1,i]+sy/2, tMatrix[1,i]+sy/2, N_POINTS_ON_EDGES ))
                    edges[:,2] = np.squeeze(np.linspace(tMatrix[2,i], tMatrix[2,i], N_POINTS_ON_EDGES ))
                    
                elif j==1:
                    edges[:,0] = np.squeeze(np.linspace(tMatrix[0,i]-sx/2, tMatrix[0,i]+sx/2, N_POINTS_ON_EDGES ))
                    edges[:,1] = np.squeeze(np.linspace(tMatrix[1,i]-sy/2, tMatrix[1,i]-sy/2, N_POINTS_ON_EDGES ))
                    edges[:,2] = np.squeeze(np.linspace(tMatrix[2,i], tMatrix[2,i], N_POINTS_ON_EDGES ))
                    
                elif j==2:
                    edges[:,0] = np.squeeze(np.linspace(tMatrix[0,i]+sx/2, tMatrix[0,i]+sx/2, N_POINTS_ON_EDGES ))
                    edges[:,1] = np.squeeze(np.linspace(tMatrix[1,i]-sy/2, tMatrix[1,i]+sy/2, N_POINTS_ON_EDGES ))
                    edges[:,2] = np.squeeze(np.linspace(tMatrix[2,i], tMatrix[2,i], N_POINTS_ON_EDGES ))  
                elif j==3:
                    edges[:,0] = np.squeeze(np.linspace(tMatrix[0,i]-sx/2, tMatrix[0,i]-sx/2, N_POINTS_ON_EDGES ))
                    edges[:,1] = np.squeeze(np.linspace(tMatrix[1,i]-sy/2, tMatrix[1,i]+sy/2, N_POINTS_ON_EDGES ))
                    edges[:,2] = np.squeeze(np.linspace(tMatrix[2,i], tMatrix[2,i], N_POINTS_ON_EDGES ))
                
                if(i>0):
                    edges = np.matmul(edges[:,:], R)
                    
                if(i==0):
                    ax.plot3D(edges[:,0], edges[:,2], edges[:,1], color='blue')
                elif(i==1): 
                    ax.plot3D(edges[:,0], edges[:,2], edges[:,1], color='red')
                elif(i==2): 
                    ax.plot3D(edges[:,0], edges[:,2], edges[:,1], color='green')                
                 
        
        ax.legend(['Camera Rotation Image 1', 'Camera Rotation Image 2', 'Camera Rotation Image 3'])     
        
        leg = ax.get_legend()
        leg.legendHandles[0].set_color('blue')
        leg.legendHandles[1].set_color('red')
        leg.legendHandles[2].set_color('green')
        
        ax.view_init(elev=36, azim=-51)
        plt.savefig(outputDir + '\\' + 'Camera Rotation_Fig1' )
        
        ax.view_init(elev=60, azim=-5)
        plt.savefig(outputDir + '\\' + 'Camera Rotation_Fig2' )
        plt.close()
        