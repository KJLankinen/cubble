"""This script plots the behaviour of the average radius against scaled time for an arbitrary amount of simulations"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from load import load_snapshots, load_input_parameters
from funcs import save_result

directory_of_file = Path(__file__).parent  # Folder location of this script
##################################################
# User defined parameters
##################################################

# Specify the paths to the folders you would like to plot
cubble_path = Path("/run/user/3206462/gvfs/smb-share:server=data.triton.aalto.fi,share=work/ulanove1/cubble")
snapshot_folder_1 = cubble_path / "data" / "06_08_2019" / "scenarios_short" / "run_1"
snapshot_folder_2 = cubble_path / "data" / "06_08_2019" / "scenarios_short" / "run_0"
snapshot_folder_3 = cubble_path / "data" / "06_08_2019" / "scenarios_short_full_flow" / "run_0"
# This is a list that contains every directory that contains snapshots to be plotted
snapshot_folders = [snapshot_folder_1, snapshot_folder_2, snapshot_folder_3]

# Titles of the individual simulations. Title should correspond to folder (!) in snapshot_folders at this stage.
# Not necessarily the same as the final graph ordering, since graphs could be rearranged
run_titles = ["No flow", "Partial Flow", "Full Flow"]

# Uncomment the following, if you would like to use data from every subdirectory of snapshot_folders_parent
# snapshot_folders_parent = directory_of_file / "data" / "12_08_2019" / "more_speeds_reduced"
# snapshot_folders = [f for f in snapshot_folders_parent.iterdir() if f.is_dir()]
# run_titles = [f.name for f in snapshot_folders]

# Determine the mode of the plot. "individual" will plot the radius development for each simulation in a separate graph.
# "trend" will show the gradients against speed in one plot
plot_mode = "individual"

input_parameters_name = "input_parameters.json"  # name of the json file that contains the input parameters
fontsize = 18  # Font size of graph elements
coarsening_factor = 1.5  # Factor of initial radius after which the coarsening is considered to have begun
##################################################

plt.rcParams.update({'font.size': fontsize})

# Empty lists that will contain information of each simulation run.
# Could be improved using dictionaries, such as
# track = dict(snapshot_numbers=[], average_radii=[], times=[], sources=[], fitted_values=[], speeds=[])

# Will be a nested list with every element belonging to one folder/simultion run and each subelement is the average
# radius for a particular snapshot
average_radii = []
times = []  # Nested list with elements corresponding to the scaled time
titles = []  # List of the final titles of the graphs
sources = []  # List of the names of the displayed sources
speeds = []  # List of flow rates for every simulation run

# Loop over every snapshot folder/simulation run. Load the snapshots, calculate the average radii for each snapshot and
# save the parameters associated with that run
for i_folder, snapshot_folder in enumerate(snapshot_folders):

    df, current_snapshot_numbers = load_snapshots(snapshot_folder)  # Load all snapshots as a pandas dataframe
    inputs = load_input_parameters(snapshot_folder / input_parameters_name)  # Load all the input parameters as dict

    # Extract input parameters for convenience
    frequency = inputs['snapshotFrequency']
    avg_init_radius = inputs['avgRad']
    kappa = inputs['kappa']
    phi = inputs["phiTarget"]
    x_speed = inputs["flowVel"]["x"]
    x_limit = inputs["flowTfr"]["x"]

    # Add the flow rate of this run
    speeds.append(x_speed)
    # Add the scaled times (snapshot number / frequency of taking snapshots)
    times.append(current_snapshot_numbers / frequency)

    df["r"] /= avg_init_radius  # Normalise the radii to the initial radius
    # Loop through each snapshot of the simulation and take the average radius for each one. Save that list of
    # average radii to the average_radii list
    average_radii.append(
        np.array([df.loc[str(i_snapshot)]["r"].mean() for i_snapshot in current_snapshot_numbers])
    )

    sources.append(f"\n\n{snapshot_folder.name}")  # Add some information of where the data came from
    # This is the title that will be displayed on top of each graph
    folder_title = run_titles[i_folder] + \
                   "\n" + rf"($flow rate={x_speed}$; $x_{{limit}}={x_limit}$" + \
                   "\n" + rf"$\kappa={kappa}; \phi={phi})$"
    titles.append(folder_title)

#############################
# Important! This will sort all the information according to the flow rate! (lowest to largest). If you do not want this
# sorting then you will need to comment out the following lines
sorted_ind = np.argsort(speeds)  # Look at speeds list and find the indices that sort the list from lowest to largest
speeds = np.array(speeds)[sorted_ind]
average_radii = np.array(average_radii)[sorted_ind]  # Sorted average radii
times = np.array(times)[sorted_ind]  # Sorted times
titles = np.array(titles)[sorted_ind]  # Sorted titles
sources = np.array(sources)[sorted_ind]  # Sorted data source descriptions
#############################


# Find the min and max of all simulations, such that the axis will be the same for all graphs for comparing
global_max_x_axis = np.concatenate(times).max()
global_min_x_axis = 1  # For log scale start at 1

global_max_y_axis = np.concatenate(average_radii).max()
global_min_y_axis = np.concatenate(average_radii).min()

# time values that will be used to calculate the fit
fit_x_values = np.linspace(global_min_x_axis, global_max_x_axis, 1000)
fitted_values = []  # output of fit y values/average radii
gradients = []  # List of gradients for every fit
for i in range(len(snapshot_folders)):

    # Radii and the time of a run. Array will be modified, so copy
    run_radii = average_radii[i].copy()
    run_times = times[i].copy()

    # only consider values for fitting where radii are larger that coarsening_factor
    coarsening_indices = (run_radii >= coarsening_factor)
    run_radii = run_radii[coarsening_indices]  # Only selected radii
    run_times = run_times[coarsening_indices]  # Only selected times

    values_to_fit = np.log10(run_radii)  # Fit should be to log values
    times_to_fit = np.log10(run_times)  # Fit should be to log values
    coefficients = np.polyfit(times_to_fit, values_to_fit, 1)  # Find fit coefficients for polynomial of order 1
    gradients.append(coefficients[0])  # Add gradient of line
    print(f"Graph {i} - Fit coefficients a * x + b: a = {coefficients[0]}, b = {coefficients[1]}")
    p = np.poly1d(coefficients)  # Create p function that is based on the calculated coefficients

    # To calculate the fitted values one needs to log the values first, since the coefficients were calculated with the
    # logged values initially. The end result should be a straight line in log scale, hence one must unlog the result
    # (use 10 **). Finally add the values to be used later
    fitted_values.append(
        10 ** p(np.log10(fit_x_values))
    )

if plot_mode == "individual":
    fig, ax = plt.subplots(1, len(snapshot_folders), figsize=(20, 12), sharey=True)  # Create as many plots as folders
    ax = ax if type(ax) is np.ndarray else [ax]  # In case want to plot only one simulation run

    # loop and create each graph
    for i, axis in enumerate(ax):
        axis.scatter(times[i], average_radii[i], marker='x', s=5, label=r"$<R>$")  # Actual data points
        # Calculated fit values
        axis.plot(fit_x_values, fitted_values[i], c="r", ls="--", label=rf"fit: $\nabla = {gradients[i]:.3f}$")
        axis.set_title(titles[i], fontsize=fontsize)

        # Set the limits that are the largest for all graphs to make comparison easier
        axis.set_xlim(global_min_x_axis, global_max_x_axis)
        axis.set_ylim(global_min_y_axis, global_max_y_axis)

        # Horizontal line from where values are considered for fitting
        axis.axhline(y=coarsening_factor, linewidth=1, color='grey', ls="--")
        if i == 0:  # set y label of the left most graph
            axis.set_ylabel(r"$<R>/<R>_{in}$", fontsize=fontsize)
        axis.set_xlabel(r"Scaled time $t^{*}$" + sources[i], fontsize=fontsize)

        # Use log scale for x and y. Comment out to see without log
        axis.set_yscale("log")
        axis.set_xscale("log")

        # Make the legend larger (especially scatter marker)
        legend = axis.legend(prop={'size': fontsize})
        legend.legendHandles[1]._sizes = [30]

elif plot_mode == "trend":
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    ax.plot(speeds, gradients, "x-")  # only useful, if have been sorted first, other wise use scatter instead of plot
    ax.set_xlabel("Flow rate")
    ax.set_ylabel(r"Gradient $\alpha$")

    pad = (speeds.max() - speeds.max()) * 0.2
    ax.set_xlim(speeds.min() - pad, speeds.max() + pad)  # Use padding for easier visualisation
else:
    raise Exception(f"Please provide a valid plot mode. Received: {plot_mode}")

# Automatically save the graph
save_name = f"{Path(__file__).stem}"  # Name of the plot
save_folder = directory_of_file / "auto_save"  # Location where to save
save_result(fig=fig, save_folder=save_folder, name=save_name, script_path=Path(__file__))  # Save figure

plt.show()