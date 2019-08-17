"""This script contains the necessary functions for loading snapshots and input parameters"""
import pandas as pd
import numpy as np
import time
import json


def load_snapshot(snapshot):
    """
    Loads all data for one snapshot csv file

    Args:
        snapshot (Path object): Path to the csv file of the snapshot

    Returns:
        pandas dataframe: Object contains all the information from the csv file

    """
    if not snapshot.exists():
        raise Exception(f"Data file does not exist: {snapshot}")

    df = pd.read_csv(snapshot)
    df.index.name = "Bubble"  # Rename index to Bubble (each row corresponds to one bubble)
    return df


def load_snapshots(snapshot_folder):
    """
    Loads all data for an entire simulation with all snapshots. Saves hdf5 file for faster repeated access

    Args:
        snapshot_folder (Path object): Path to the folder that contains all the csv snapshot files

    Returns:
        pandas dataframe: Object contains all snapshots of the run. First index is the snapshot number (str),
                          second index is the bubble number (int) [not same as ID!]
        numpy array: array of all snapshot numbers in dataframe

    """
    start_snaps_load = time.time()  # Track the start time
    snapshot_pattern = 'snapshot.csv.[0-9]*'  # pattern of snapshot csv files
    # When all snapshots are loaded will save the dataframe with this name for faster access
    h5_name = 'all_snapshots.hdf5'
    h5_data_path = snapshot_folder / h5_name  # path where HDF5 file will be saved

    # If HDF5 file does not exist load snapshots one by one from csv files
    if not h5_data_path.exists():
        snapshots = np.asarray(sorted(snapshot_folder.glob(snapshot_pattern)))  # Find all csv snapshot files
        if len(snapshots) == 0:
            raise Exception(f"No snapshots found in {snapshot_folder.resolve()}")
        print(f"Loading csv snapshots ({len(snapshots)} total) from:{snapshot_folder.resolve()}")

        # Need to first sort the the snapshots according to their number, otherwise loaded in wrong order
        snapshot_numbers = [int(str(file).split(".")[-1]) for file in snapshots]  # Number of each snapshot file
        sorted_indices = np.argsort(snapshot_numbers)  # Indices that sort them form low to high
        snapshots = snapshots[sorted_indices]  # New sorted snapshots list
        snapshot_names = [s.name.split(".")[-1] for s in snapshots]  # Number of snapshot

        snapshots_dfs = []  # Will collect all individual snapshots here

        for i, snapshot in enumerate(snapshots):
            start_snaps_csv_load = time.time()
            snapshot_df = load_snapshot(snapshot)  # Load the snapshot as dataframe
            snapshots_dfs.append(snapshot_df)
            print(f"Loaded: {snapshot.name} ({i} of {len(snapshots) - 1}) in {time.time() - start_snaps_csv_load:.2f}s")

        # Merge all snapshots into one object and label them by their name/number
        all_snapshots_df = pd.concat(snapshots_dfs, keys=snapshot_names)
        all_snapshots_df.index.names = ["Snapshot", "Bubbles"]  # The first index is the snapshot. Second is the bubble
        print(f"Loaded snapshots from:{snapshot_folder.resolve()}")
        print(f"Saving snapshots to HDF5 format for faster repeated access:{h5_data_path.resolve()}")
        all_snapshots_df.to_hdf(h5_data_path.resolve(), key='df', mode='w')  # Save the dataframe for faster access
        print(f"Saved")
    else:  # Data has already been prepared before and thus just load the whole file
        print(f"Loading data from HDF5 file: {h5_data_path.resolve()}")
        all_snapshots_df = pd.read_hdf(h5_data_path.resolve(), 'df')

    print(f"Loaded snapshots in {time.time() - start_snaps_load:.2f}s")
    # get the snapshot numbers based on the first index of the dataframe
    snapshot_numbers = np.array([int(s) for s in all_snapshots_df.index.get_level_values(0).unique().values])
    return all_snapshots_df, snapshot_numbers


def load_input_parameters(input_parameter_path):
    """
    Loads the input parameters of a simulation run

    Args:
        input_parameter_path (Path object): Path to the input_parameters.json file

    Returns:
        dict: Dictionary where key is input parameter and value is the input parameter value

    """
    with open(input_parameter_path) as input_file:
        return json.load(input_file)

