"""This script converts csv snapshot files to vtu files"""
import argparse
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from pyevtk.hl import pointsToVTK


########################################################################################################################
# User adjustable variables
########################################################################################################################

csv_pattern = 'snapshot.csv.[0-9]*'  # Pattern of csv snapshot files
default_output_folder_name = "vtu_snapshots"  # Name of the output folder of the vtu files
run_folder_pattern = 'run_[0-9]*'  # Pattern of sub folders of a simulation that contain the snapshots
logging.basicConfig(level=logging.WARN)  # logging.INFO or logging.DEBUG or logging.WARNING

########################################################################################################################
directory_of_this_script = Path(__file__).parent  # Current script file folder path
info = logging.info; debug = logging.debug; warn = logging.warning

# Some possible user provided arguments and the description
parser = argparse.ArgumentParser(
    description='This script converts csv snapshots in a given directory to vtu files'
)

parser.add_argument(
    '--snapshot_dir',
    type=str,
    help='Location of folder containing the snapshots to be converted'
)

parser.add_argument(
    '--save_dir_name',
    type=str,
    default=default_output_folder_name,
    help='(Optional) Name of directory to be created in --snapshot_dir where all converted snapshots will be saved'
)

parser.add_argument(
    '-s',
    action='store_true',
    help=f'(Optional flag) If set will attempt to find all folders with pattern "{run_folder_pattern}" '
    f'(or specified with --sub_folder_pattern) in --snapshot_dir and convert all snapshots in the found folders'
)

parser.add_argument(
    '--sub_folder_pattern',
    type=str,
    default=run_folder_pattern,
    help=f'(Optional) If -s is set this is the pattern that is used to recognise subfolders in --snapshot_dir '
    f'that contain snapshots. Default: "{run_folder_pattern}"'
)

parser.add_argument(
    '-i',
    action='store_true',
    help=f'(Optional) Change the level of output that is printed to "INFO": print all messages with useful information, '
    f'otherwise only critical messages will be printed.'
)

# Save the user given parameters for easier access
convert_namespace = parser.parse_args()
run_folder_pattern = str(convert_namespace.sub_folder_pattern)
snapshot_dir = convert_namespace.snapshot_dir
find_subfolders = convert_namespace.s
save_dir_name = convert_namespace.save_dir_name
if convert_namespace.i:
    logging.getLogger().setLevel(logging.INFO)  # Setting the level of print output


def convert_run(snapshot_folder, output_folder, snapshot_pattern):
    """
    Converts all csv snapshots in a folder to vtu files and saves them in a separate folder.

    Args:
        snapshot_folder (Path object): Path to the folder where the csv snapshots are located
        output_folder (Path object): Path to the folder where the vtu snapshots will be saved

    """
    # Find all csv snapshots according to the pattern
    snapshot_paths = np.asarray(sorted(snapshot_folder.glob(snapshot_pattern)))

    if snapshot_paths.size == 0:  # Check, if any snapshots are availible
        warn(f"No snapshot to convert found in {snapshot_folder}")
    else:
        if not output_folder.exists():  # Create output folder, if does not exist yet
            output_folder.mkdir(exist_ok=False)
            info(f"Vtu converted save directory created in {output_folder.resolve()}")
        else:
            raise Exception(f"Save dir already exists: {save_dir.resolve()}. Abort")

        info(f"Output directory {output_folder} was created")
        start = time.time()

        # Load the first snapshot to determine the scalars and vectors
        data = pd.read_csv(snapshot_paths[0], delimiter=",")
        column_names = data.columns.values

        # vector are searched for in the column names. Parameters that end in x, y and z are considered to be components
        # of a vector. E.g. if in column names there is vx, vy and vz, v is saved as a vector
        vector_names = np.unique(
            [
                c[:-1] for c in column_names
                if (c[:-1] != "" and
                    c[:-1] + "x" in column_names and
                    c[:-1] + "y" in column_names and
                    c[:-1] + "z" in column_names)
            ]
        )
        # everything that is not a vector is considered a scalar, except for x, y, z, since these need to be saved
        # separately
        scalar_names = np.array(
            [
                c for c in column_names
                if (c[:-1] not in vector_names and
                    c != "x" and
                    c != "y" and
                    c != "z")
            ]
        )

        # In turn load each snapshot
        for snapshot_path in snapshot_paths:
            start_snap = time.time()
            data = pd.read_csv(snapshot_path, delimiter=",")  # Load snapshot

            # create dictionary with all scalars and separately all vectors
            scalars = {sc: data[sc].values for sc in scalar_names}
            vectors = {vc: (data[vc + "x"].values, data[vc + "y"].values, data[vc + "z"].values) for vc in vector_names}
            d = {**scalars, **vectors}  # Merge the two dictionaries

            i = str(snapshot_path).split(".")[-1]  # Get the snapshot number from the filename
            # Create a name that is descriptive when opening in Paraview
            save_path = output_folder / f"snapshot_{snapshot_folder.parents[1].stem}_{snapshot_folder.parents[0].stem}_{snapshot_folder.stem}_{i}"
            # Write a VTU binary file from the snapshot data.
            # The structure is: pointsToVTK(filename/path, x values, y values, z values, rest of data)
            saved_to = pointsToVTK(str(save_path), data["x"].values, data["y"].values, data["z"].values, data=d)
            info(f"Converted snapshot {snapshot_path} to {saved_to} in {time.time() - start_snap}s")

        diff = time.time() - start
        info(f"Converted {snapshot_paths.size} snapshots in {diff:.2f}s ({snapshot_paths.size / diff:.2f} snapshots/s)")

#if len(sys.argv) > 1:
#    raise Exception("Wrong arguments given. Use --help to find list of arguments")


# If user supplied a directory that contains snapshots
if snapshot_dir:
    user_given_snapshot_directory = snapshot_dir
    snapshot_folder = Path(user_given_snapshot_directory)

    if not snapshot_folder.exists():  # Check if the user given dir exists
        raise Exception(f"Snapshot directory {snapshot_folder.resolve()} does not exist. Please give absolute path.")

    try:
        if find_subfolders:  # if user set the option to look in subfolders
            # will find the sub folder that should contain snapshots, according to the pattern given
            sub_folders = sorted(snapshot_folder.glob(run_folder_pattern))

            if len(sub_folders) == 0:
                warn(f"No matching subfolders with pattern {run_folder_pattern} found in {snapshot_folder.resolve()}")

            for sub_folder in sub_folders:  # Try to convert in each sub dir separately
                save_dir = sub_folder / save_dir_name
                convert_run(sub_folder, save_dir, csv_pattern)
        else:  # User gave the exact location where the files should be and will not look in sub directories
            save_dir = snapshot_folder / save_dir_name
            convert_run(snapshot_folder, save_dir, csv_pattern)

    except Exception as e:
        warn(f"Could not convert files in {snapshot_folder}:{e}")

else:
    # Assumes that user wants to convert files relative to this script.
    # Will try to convert everything according to the structure:
    # - this script
    # - data
    #   - 01_01_2099  # All dates of this sort
    #       - any folder name
    #           - run_99 # or whatever is given for run_folder_pattern
    #               - csv snapshots
    # This will done for all folders in data
    data_directory = directory_of_this_script / "data"

    warn("No arguments given, assuming that everything in data folder should be converted "
          "(If this is accidental use --help to see the possible arguments)")

    warn(f"\nAre you sure you want to find all possible snapshots in {data_directory.resolve()} and convert them "
          f"\n(this is NOT recommended and may take a long time)? [y/n]:")
    x = input()
    while x != "y" and x != "n":
        warn("Please enter y or n")
        x = input()

    if x == "y":
        sub_folders = sorted(data_directory.glob("[0-9][0-9]_[0-9][0-9]_[0-9][0-9][0-9][0-9]"))

        for sub_folder in sub_folders:
            sub_sub_folders = sorted(sub_folder.glob("*"))
            for sub_sub_folder in sub_sub_folders:
                if sub_folder.is_dir():
                    run_folder_paths = np.asarray(sorted(sub_sub_folder.glob(run_folder_pattern)))
                    for run_folder in run_folder_paths:
                        output_folder = run_folder / default_output_folder_name
                        if not output_folder.exists():
                            try:
                                convert_run(run_folder, output_folder, csv_pattern)
                            except Exception as e:
                                warn(f"Could not convert files in {run_folder}:{e}")