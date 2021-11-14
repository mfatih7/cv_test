
import numpy as np
import matplotlib.pyplot as plt


ax = plt.axes(projection="3d")

x = np.linspace(1, 100, 100)
y = np.linspace(1, 10, 100)
z = np.linspace(1, 1, 100)

ax.plot3D(x,y,z)

ax.plot3D(y,x,z)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()

# # mpl.rcParams['legend.fontsize'] = 10

# # fig = plt.figure()
# # fig = fig.gca(projection='3d')

# x = np.linspace(1, 100, 100)
# y = np.linspace(1, 10, 100)
# z = np.linspace(1, 1, 100)

# plt.plot(x, y, z)
# # ax.legend()


plt.savefig('filename.png')