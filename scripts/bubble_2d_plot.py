import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import *

def drawSphere(xCenter, yCenter, zCenter, r):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = r * np.outer(np.cos(u), np.sin(v)) + xCenter
    y = r * np.outer(np.sin(u), np.sin(v)) + yCenter
    z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + zCenter
    
    return (x,y,z)

def circles(x, y, s, c='b', vmin=None, vmax=None, **kwargs):
    """
    Make a scatter of circles plot of x vs y, where x and y are sequence 
    like objects of the same lengths. The size of circles are in data scale.

    Parameters
    ----------
    x,y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, ) 
        Radius of circle in data unit.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence 
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)  
        `c` can be a 2-D array in which the rows are RGB or RGBA, however. 
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls), 
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    a = np.arange(11)
    circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.collections import PatchCollection

    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None
    if 'fc' in kwargs: kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs: kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs: kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs: kwargs.setdefault('linewidth', kwargs.pop('lw'))

    patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    if c is not None:
        plt.sci(collection)
    return collection

def plot_2d(n):
    fig = plt.figure()
    axes = []
    for i in range(n):
        filename = "data/snapshot" + str(i + 5) + ".dat"

        with open(filename) as file:
            head = [next(file) for x in xrange(11)]

        lbb = np.fromstring(head[-2], count=3, sep=',')
        tfr = np.fromstring(head[-1], count=3, sep=',')
            
        data = np.loadtxt(filename, delimiter=',', skiprows=11)
            
        x = lbb[0] + data[:,0] * (tfr - lbb)[0]
        y = lbb[1] + data[:,1] * (tfr - lbb)[1]
        r = data[:,3]
            
        axes.append(fig.add_subplot(np.ceil(n / 2.0), np.clip(np.ceil(n / 1.9), 1, 2), i + 1, aspect='equal'))
        out = circles(x, y, r, alpha=0.3)
    
    plt.show()

def plot_3d():
    filename = "data/snapshot2.dat"

    with open(filename) as file:
        head = [next(file) for x in xrange(11)]
    lbb = np.fromstring(head[-2], count=3, sep=',')
    tfr = np.fromstring(head[-1], count=3, sep=',')
    
    data = np.loadtxt(filename, delimiter=',', skiprows=11)

    x = lbb[0] + data[:,0] * (tfr - lbb)[0]
    y = lbb[1] + data[:,1] * (tfr - lbb)[1]
    z = lbb[2] + data[:,2] * (tfr - lbb)[2]
    r = data[:,3]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    for (xi, yi, zi, ri) in zip(x, y, z, r):
        (xs, ys, zs) = drawSphere(xi, yi, zi, ri)
        ax.plot_surface(xs, ys, zs)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    plt.show()
    
def main():
    plot_2d(4)

if __name__ == "__main__":
    main()