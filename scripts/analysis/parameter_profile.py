"""This script calculates a profile of any parameter for an entire simulation"""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

from load import load_snapshots
from funcs import save_result


directory_of_file = Path(__file__).parent


def volume_fraction(radii):
    """
    Calculates the volume fraction of spheres occupying an area area_box. Overlap between spheres is counted twice

    Args:
        radii (numpy array): Array of sphere radii that occupy an area

    Returns:
        float: Volume fraction occupied. Overlap is not separately handled. E.g two spheres of same size at the same
               position with each sphere having a volume of 1 will count as volume of two spheres even though they are
               physically occupying the same space. So it will be counted as total volume of 2 instead of 1

    """
    total_bubble_area = np.sum(np.pi * (radii ** 2))
    return total_bubble_area / area_box


##################################################
# User defined parameters
##################################################
parameter_to_apply_statistics = "vx"  # Parameter which be used as the basis for the statistics
cubble_path = Path("/run/user/3206462/gvfs/smb-share:server=data.triton.aalto.fi,share=work/ulanove1/cubble")
snapshot_folder = cubble_path / "data" / "31_07_2019" / "multiple_speeds" / "speed3"  # Folder with simulation snapshots
binnum = 30  # Number of bins to use
# Defines the function that will be used to calculate the bin values. Use "mean" to get the mean value or any other
# user defined function
statistic_to_apply = "mean"
# defines which snapshot to start from for the analysis. Careful this is the snapshot name number (!), i.e. the number
# that was part of the original filename. It is NOT an index. In most cases however these will be equivalent
start_at_snap = 0
# Set to True if want to normalise all values in a snapshot to the average of the snapshot.
# Ensures that values that are very large compared to values at other times dominate the result
normalise_to_average = True
label = rf"$<{parameter_to_apply_statistics}>$"  # x label. Can choose freely
title = f"{'Normalised ' if normalise_to_average else ''}{label}"  # title. Can choose freely

# Percentage of x range that is considered in the analysis from the mid point to each side.
# Should be between 0 and 0.5. E.g. if x range is [0, 10],
# then percentage_from_midpoint=0.15 would consider all bubbles that lie between x = 3.5 and 6.5
# Setting to 0.5 considers all points
percentage_from_midpoint = 0.5

fontsize = 18
##################################################
# Cange the axis labelling size
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)

df, snapshot_numbers = load_snapshots(snapshot_folder)
df, snapshot_numbers = df[str(start_at_snap):], snapshot_numbers[start_at_snap:]

range_x = df["x"].max() - df["x"].min()
# Mid point of the x range. Careful will not be correct, if size of simulation box changes from snapshot to snapshot
mid_x = range_x / 2
# Only select bubbles that lie within the specified range from the mid point
df = df[
    (df["x"] >= mid_x - range_x * percentage_from_midpoint) &
    (df["x"] <= mid_x + range_x * percentage_from_midpoint)
]


# Only necessary, if wanting to calculate the gas density with the volume_fraction function
y_bin_width = (df["y"].max() - df["y"].min()) / binnum
area_box = y_bin_width * range_x

averages_of_snapshots = []
for i in snapshot_numbers:
    snap_df = df.loc[str(i)]  # Load the data for each snapshot separately
    parameter_values = snap_df[parameter_to_apply_statistics]  # Select the user defined column
    # Here the y coordinate of a bubble is used to sort them into bins. Then e.g. the mean of all 'vx' in a bin is
    # calculated to get an idea of how the bubbles are moving in x direction in one y slice segment. The width of the
    # y slice is determined by the value of bins, i.e. reduce bins to get wider slices.
    # The values of the first argument of binned_statistic is sorted into bins. the function statistic_to_apply
    # determines what will be done with all the values in one bin. It takes the second argument as its input and returns
    # one number. This value is the final value of the bin. E.g. using statistic_to_apply -> "mean" calculates the mean
    # of all values that were assigned to an individual bin by the first argument of binned_statistic
    # bin_values are the final values of each bin; can be not mean, depending on what statistic_to_apply is defined
    # bin_edges gives the edges of the bins and binnumber the bin number used
    bin_values, bin_edges, binnumber = stats.binned_statistic(snap_df["y"], parameter_values, statistic_to_apply, bins=binnum)
    bin_width = (bin_edges[1] - bin_edges[0])  # width of one bin
    bin_centers = bin_edges[1:] - bin_width/2  # mid point of bins
    # If normalise_to_average is True will normalise each value to the average. Useful, if values vary wildly in between
    # snapshots and one is only interested in the relative behaviour within one snapshot
    relative_average_r = bin_values / np.average(bin_values) if normalise_to_average else bin_values
    averages_of_snapshots.append(relative_average_r)

# For each bin calculated the average over all snap shots.
# If averages_of_snapshots is (snapshots * bins)
# [[1, 2, 2]
#  [2, 2, 0]
#  [3, 2, 1]]
# then the result will be [2, 2, 1]
average_over_whole_run = np.average(averages_of_snapshots, axis=0)

fig, ax = plt.subplots(1, 1, figsize=(8, 20))
origin_str = f"{snapshot_folder.parents[1].stem}/{snapshot_folder.parents[0].stem}/{snapshot_folder.stem}"

fig.suptitle(origin_str)  # Meaningful title for easier tracking of data source later

ax.plot(average_over_whole_run, bin_centers, 'o-')  # Plot with y axis representing the original y values
ax.set_title(title, fontsize=fontsize)
ax.set_xlabel(label, fontsize=fontsize)
ax.set_ylabel("y", fontsize=fontsize)


# Automatically save the figure
save_name = f"{snapshot_folder.parents[1].stem}_{snapshot_folder.parents[0].stem}_{snapshot_folder.stem}-{Path(__file__).stem}"
save_folder = directory_of_file / "auto_save"
save_result(fig=fig, save_folder=save_folder, name=save_name, script_path=Path(__file__))

plt.show()