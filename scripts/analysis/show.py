import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.colors

from load import df
from funcs import vorticity

x = df["x"]
y = df["y"]
r = df["r"]
vx = df["vx"]
vy = df["vy"]
path = df["path"]
dist = df["dist"]

average_r = np.average(r) * 0.5
#n = np.sqrt(df.shape[0]) * 5

delta_x = average_r #abs(max(x) - min(x))/n
delta_y = average_r #abs(max(y) - min(y))/n

n = int(abs(max(x) - min(x)) / average_r)
#n = 300
#n = np.sqrt(df.shape[0])
grid_x, grid_y = np.mgrid[min(x):max(x):n * 1j, min(y):max(y):n * 1j]

grid_vx = griddata(np.c_[x, y], vx, (grid_x, grid_y), method='cubic')
grid_vy = griddata(np.c_[x, y], vy, (grid_x, grid_y), method='cubic')

p = np.vstack([grid_x.ravel(), grid_y.ravel()])
p2 = np.array(list(zip(p[0, :], p[1, :])))

vxx = np.vstack([grid_vx.ravel(), grid_vx.ravel()])
v2x = np.array(list(zip(vxx[0, :], vxx[1, :])))

vyy = np.vstack([grid_vy.ravel(), grid_vy.ravel()])
v2y = np.array(list(zip(vyy[0, :], vyy[1, :])))

fig, ax = plt.subplots(2, 3, figsize=(20, 10))


cmap = plt.cm.RdYlBu
norm = matplotlib.colors.Normalize(vmin=min(r), vmax=max(r))
ax[0, 0].scatter(x, y, s=0)
for i in range(x.size):
    disk = plt.Circle((x[i], y[i]), radius=r[i], facecolor=cmap(norm(r[i])), edgecolor="black", linewidth=0.1)
    ax[0, 0].add_artist(disk)
ax[0, 0].set_title("Size")
ax[0, 0].set_xlabel("x")
ax[0, 0].set_ylabel("y")

ax[1, 0].quiver(x, y, vx, vy, np.hypot(vx, vy))
ax[1, 0].set_title("Velocity")

ax[1, 1].quiver(p2[:, 0], p2[:, 1], v2x[:, 0], v2y[:, 0], np.hypot(v2x[:, 0], v2y[:, 0]))
ax[1, 1].set_title("Velocity Interpolation")

grid_vorticity4 = vorticity(VX=grid_vx, VY=grid_vy, delta_x=delta_x, delta_y=delta_y, acc=4)
l = ax[1, 2].imshow(np.flip(np.flip(grid_vorticity4.T), 1), cmap='jet', aspect='equal', extent=[min(x), max(x), min(y), max(y)])
ax[1, 2].set_title("Vorticity")

norm2 = matplotlib.colors.Normalize(vmin=min(path), vmax=max(path))
ax[0, 1].scatter(x, y, s=0)
for i in range(x.size):
    disk = plt.Circle((x[i], y[i]), radius=r[i], facecolor=cmap(norm2(path[i])), edgecolor="black", linewidth=0.1)
    ax[0, 1].add_artist(disk)
ax[0, 1].set_title("Path")

norm3 = matplotlib.colors.Normalize(vmin=min(dist), vmax=max(dist))
ax[0, 2].scatter(x, y, s=0)
for i in range(x.size):
    disk = plt.Circle((x[i], y[i]), radius=r[i], facecolor=cmap(norm2(dist[i])), edgecolor="black", linewidth=0.1)
    ax[0, 2].add_artist(disk)
ax[0, 2].set_title("Distance")

for a in ax.flatten():
    a.set(aspect='equal')
    a.set_xlim(min(x), max(x))
    a.set_ylim(min(y), max(y))

plt.show()