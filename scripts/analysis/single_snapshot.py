"""
With this script can view different parameters of a single snapshot. Also calculates vorticity
"""
# This script could be improved, by making the vorticity calculation simpler
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.colors
from matplotlib.colors import SymLogNorm
from pathlib import Path
import sys

from load import load_snapshot
from funcs import vorticity

directory_of_file = Path(__file__).parent
#######################################
# User defined parameters
#######################################
cubble_path = Path("/run/user/3206462/gvfs/smb-share:server=data.triton.aalto.fi,share=work/ulanove1/cubble")
snapshot = cubble_path / "data" / "06_08_2019" / "scenarios_short" / "run_0" / "snapshot.csv.150"
#######################################
df = load_snapshot(snapshot)

x = df["x"]
y = df["y"]
r = df["r"]
vx = df["vx"]
vy = df["vy"]
path = df["path"]
energy = df["energy"]

# Distance between point for interpolation
delta_x = np.average(r) * 0.5
delta_y = delta_x

n = int(abs(max(x) - min(x)) / delta_x)  # Number of points in each direction
grid_x, grid_y = np.mgrid[min(x):max(x):n * 1j, min(y):max(y):n * 1j]  # Grid of x and y coordinates

# Use cubic spline interpolation to interpolate for the points of grid_x and grid_x with the values at x and y
# Interpolation happens for each component separately
grid_vx = griddata(np.c_[x, y], vx, (grid_x, grid_y), method='cubic')
grid_vy = griddata(np.c_[x, y], vy, (grid_x, grid_y), method='cubic')

# Converts meshgrids into list of points; here the positions
p = np.vstack([grid_x.ravel(), grid_y.ravel()])
p2 = np.array(list(zip(p[0, :], p[1, :])))

# Converts meshgrids into list of points; here the vx velocities
vxx = np.vstack([grid_vx.ravel(), grid_vx.ravel()])
v2x = np.array(list(zip(vxx[0, :], vxx[1, :])))

# Converts meshgrids into list of points; here the vy velocities
vyy = np.vstack([grid_vy.ravel(), grid_vy.ravel()])
v2y = np.array(list(zip(vyy[0, :], vyy[1, :])))

# Now each p2 point has corresponding v2x vx velocity and v2y vy velocity

# Plot will be 6 figures.
# Top 3: Coloured by radius, path and energy.
# Bottom 3: velocity field, interpolated velocity and vorticity
fig, ax = plt.subplots(2, 3, figsize=(20, 10))
origin_str = f"{snapshot.parent.parent.name}/{snapshot.parent.name}/{snapshot.name}"
fig.suptitle(origin_str)

# Graph coloured by radius
cmap = plt.cm.RdYlBu
norm_radius = matplotlib.colors.Normalize(vmin=min(r), vmax=max(r))  # Assign each radius a different color
ax[0, 0].scatter(x, y, s=0)  # Centre of bubble
for i in range(x.size):  # Draw each sphere individually
    disk = plt.Circle((x[i], y[i]), radius=r[i], facecolor=cmap(norm_radius(r[i])), edgecolor="black", linewidth=0.1)
    ax[0, 0].add_artist(disk)
ax[0, 0].set_title("Size")
ax[0, 0].set_xlabel("x")
ax[0, 0].set_ylabel("y")

# Graph path
norm_path = matplotlib.colors.Normalize(vmin=min(path), vmax=max(path))
ax[0, 1].scatter(x, y, s=0)
for i in range(x.size):
    disk = plt.Circle((x[i], y[i]), radius=r[i], facecolor=cmap(norm_path(path[i])), edgecolor="black", linewidth=0.1)
    ax[0, 1].add_artist(disk)
ax[0, 1].set_title("Path")

# Graph energy
norm_energy = matplotlib.colors.Normalize(vmin=min(energy), vmax=max(energy))
ax[0, 2].scatter(x, y, s=0)
for i in range(x.size):
    disk = plt.Circle((x[i], y[i]), radius=r[i], facecolor=cmap(norm_energy(energy[i])), edgecolor="black", linewidth=0.1)
    ax[0, 2].add_artist(disk)
ax[0, 2].set_title("Energy")

# Graph measured velocity field
a = ax[1, 0].quiver(x, y, vx, vy, np.hypot(vx, vy), scale=0.3, width=0.005)
ax[1, 0].set_title("Velocity")
fig.colorbar(a, ax=ax[1, 0])  # Colorbar

# Graph interpolated velocity field
ax[1, 1].quiver(p2[:, 0], p2[:, 1], v2x[:, 0], v2y[:, 0], np.hypot(v2x[:, 0], v2y[:, 0]))
ax[1, 1].set_title("Velocity Interpolation")

# Graph vorticity
grid_vorticity = vorticity(VX=grid_vx, VY=grid_vy, delta_x=delta_x, delta_y=delta_y, acc=4)
ax[1, 2].imshow(np.flip(np.flip(grid_vorticity.T), 1), cmap='jet', aspect='equal', extent=[min(x), max(x), min(y), max(y)],
                    #norm=SymLogNorm(linthresh=0.00001, vmin=np.nanmin(grid_vorticity), vmax=np.nanmax(grid_vorticity))
                    )
ax[1, 2].set_title("Vorticity")

# Make sure all graphs have the same aspect ratio and same x, y limits
for a in ax.flatten():
    a.set(aspect='equal')
    a.set_xlim(min(x), max(x))
    a.set_ylim(min(y), max(y))

plt.show()