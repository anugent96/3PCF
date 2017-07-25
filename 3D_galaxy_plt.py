"""
https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html#surface-plots
"""

from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

"""
# For Cutsky Plots
x = np.load('Desktop/3PCF/CutskyN1_x.npy')
y = np.load('Desktop/3PCF/CutskyN1_y.npy')
z = np.load('Desktop/3PCF/CutskyN1_z.npy')

#plt.plot(x, y,'ro')
#plt.show()


# For Uncut Box Plots
x = np.load('Desktop/UncutBox1_x.npy')
y = np.load('Desktop/UncutBox1_y.npy')
z = np.load('Desktop/UncutBox1_z.npy')
"""

# For Uncut Rotated Box Plots
x = np.load('Desktop/UncutBox1_x_rot.npy')
y = np.load('Desktop/UncutBox1_y_rot.npy')
z = np.load('Desktop/UncutBox1_z_rot.npy')

x = np.array(x)
y = np.array(y)
z = np.array(z)
                                       
ax.scatter(x, y, z, zdir='z')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Uncut Box 1, 1st 1000 Obj. Positions')

plt.show()