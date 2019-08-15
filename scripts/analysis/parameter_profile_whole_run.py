"""
This script visualises a parameter through out a simulation that is binned along a particular axis.
This is shown in form of a 3D graph and a heat map
"""
import numpy as np
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from pathlib import Path

from load import load_snapshots, load_input_parameters
from funcs import save_result

directory_of_file = Path(__file__).parent
#######################################
# User defined parameters
##################################################
parameter_to_apply_statistics = "vx"  # Parameter which be used as the basis for the statistics
axis = "y"  # Axis along which the values will be sorted into bins
statistic_title = f"Average '{parameter_to_apply_statistics}'"  # Descriptive title of result
cubble_path = Path("/run/user/3206462/gvfs/smb-share:server=data.triton.aalto.fi,share=work/ulanove1/cubble")
snapshot_folder = cubble_path / "data" / "31_07_2019" / "multiple_speeds" / "speed3"
# Defines the function that will be used to calculate the bin values. Use "mean" to get the mean value or any other
# user defined function
statistics_function = "mean"
input_parameters_name = "input_parameters.json"  # name of the json file that contains the input parameters
binnum = 25  # Number of bins to use
time_label = r"Scaled time $t^{*}$"
#######################################

df, snapshot_numbers = load_snapshots(snapshot_folder)
snapshots_length = len(snapshot_numbers)

# Getting simulation parameter from file
frequency = load_input_parameters(snapshot_folder / input_parameters_name)["snapshotFrequency"]

bin_values_whole_run = []
for snapshot_number in snapshot_numbers:
    snap_df = df.loc[str(snapshot_number)]  # Load the data for each snapshot separately
    # Here the axis coordinate of a bubble is used to sort them into bins. Then e.g. the mean of all 'vx' in a bin is
    # calculated to get an idea of how the bubbles are moving in x direction in one y slice segment. The width of the
    # y/x slice is determined by the value of bins, i.e. reduce bins to get wider slices.
    # The values of the first argument of binned_statistic is sorted into bins. the function statistic_to_apply
    # determines what will be done with all the values in one bin. It takes the second argument as its input and returns
    # one number. This value is the final value of the bin. E.g. using statistic_to_apply -> "mean" calculates the mean
    # of all values that were assigned to an individual bin by the first argument of binned_statistic
    # bin_values are the final values of each bin; can be not mean, depending on what statistic_to_apply is defined
    # bin_edges gives the edges of the bins and binnumber the bin number used
    bin_values, bin_edges, binnumber = stats.binned_statistic(snap_df[axis], snap_df[parameter_to_apply_statistics],
                                                              statistics_function, bins=binnum)
    bin_values_whole_run.append(bin_values)

# Each row contains a list of the bin values for a bin
bin_values_whole_run = np.asarray(bin_values_whole_run)

bin_width = (bin_edges[1] - bin_edges[0])  # width of one bin
bin_centers = bin_edges[1:] - bin_width/2  # mid point of bins

# Will create two separate graphs
fig = plt.figure(figsize=(24, 12))
origin_str = f"{snapshot_folder.parents[1].stem}/{snapshot_folder.parents[0].stem}/{snapshot_folder.stem}"
fig.suptitle(origin_str)  # Meaningful title for easier tracking of data source later

# First plot the 3D visualisation
ax = fig.add_subplot(1, 2, 1, projection='3d')
t = snapshot_numbers / frequency  # list of all times
bv = bin_centers  # list of the centres of the bins
BV, T = np.meshgrid(bv, t)  # create coordinate grid
ax.plot_surface(BV, T, bin_values_whole_run, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel(axis)
ax.set_ylabel(time_label)
ax.set_zlabel(statistic_title)

# Second plot will be a heat map for an easier overview
ax = fig.add_subplot(1, 2, 2)

# uncomment one(!) of the norm parameters to either see a log scale or for both pos and neg values a symmetric log scale
heatmap = ax.imshow(bin_values_whole_run, cmap='coolwarm',
                    #norm=colors.LogNorm(vmin=bin_means_all.min(), vmax=bin_means_all.max()),
                    #norm=colors.SymLogNorm(linthresh=0.00001, vmin=bin_means_all.min(), vmax=bin_means_all.max()),
                    extent=[df[axis].min(), df[axis].max(), snapshot_numbers.min() - 0.5, snapshot_numbers.max() + 0.5])
ax.set_ylabel(time_label)
ax.set_xlabel(axis)
clb = fig.colorbar(heatmap)  # Create color bar next to heatmap
clb.ax.set_title(statistic_title)

# Automatically save the figure
save_name = f"{snapshot_folder.parents[1].stem}_{snapshot_folder.parents[0].stem}_{snapshot_folder.stem}-{Path(__file__).stem}"
save_folder = directory_of_file / "auto_save"
save_result(fig=fig, save_folder=save_folder, name=save_name, script_path=Path(__file__))

plt.show()