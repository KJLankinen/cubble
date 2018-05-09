import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0

def drawSphere(xCenter, yCenter, zCenter, r):
    #draw sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x=np.cos(u)*np.sin(v)
    y=np.sin(u)*np.sin(v)
    z=np.cos(v)

    # shift and scale sphere
    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter
    
    return (x,y,z)

def main():
    num_points = 1000
    num_files = 1000
    final_x = np.zeros((num_points))
    final_y = np.zeros((num_points))
    final_z = np.zeros((num_points))
    final_r = np.zeros((num_points))
    final_c = np.zeros((num_points, 3))
    
    j = 0
    for i in range(num_files):
        path = "./data/bubble_refs_" + str(i) + ".dat"
        if (is_non_zero_file(path)):
            data = np.loadtxt(path, delimiter=',')

            x = np.empty((1))
            y = np.empty((1))
            z = np.empty((1))
            r = np.empty((1))
            c = np.empty((1, 3))

            data_length = 1
            if (len(data.shape) > 1):
                x = data[:,0]
                y = data[:,1]
                z = data[:, 2]
                r = data[:,3]
                c = np.ones((len(r), 3))
                data_length = len(x)
            else:
                x = data[0]
                y = data[1]
                z = data[2]
                r = data[3]
                c = np.ones((1, 3))

            c *= np.array([(i % 10) / 10.0, (i % 100 / 10) / 10.0, (i / 100) / 10.0])

            final_x[j : j + data_length] = x
            final_y[j : j + data_length] = y
            final_z[j : j + data_length] = z
            final_r[j : j + data_length] = r
            final_c[j : j + data_length] = c
            j = j + data_length

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for (xi,yi,zi,ri, ci) in zip(final_x, final_y, final_z, final_r, final_c):
        (xs,ys,zs) = drawSphere(xi,yi,zi,ri)
        ax.plot_wireframe(xs, ys, zs, color=ci)
        
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
            
    plt.show()

if __name__ == "__main__":
    main()
