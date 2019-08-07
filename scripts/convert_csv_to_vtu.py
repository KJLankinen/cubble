from pyevtk.vtk import VtkGroup
from pyevtk.hl import pointsToVTK
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import time
import argparse


########################################################################################################################
# User adjustable variables
########################################################################################################################

csv_pattern = 'snapshot.csv.[0-9]*'
default_output_folder_name = "vtu_snapshots"
run_folder_pattern = 'run_[0-9]*'
logging.basicConfig(level=logging.WARN)  # logging.INFO or logging.DEBUG or logging.WARNING

########################################################################################################################
directory_of_this_script = Path(__file__).parent  # Current script file folder path
data_directory = directory_of_this_script / "data"
info = logging.info; debug = logging.debug; warn = logging.warning


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
convert_namespace = parser.parse_args()
run_folder_pattern = str(convert_namespace.sub_folder_pattern)
snapshot_dir = convert_namespace.snapshot_dir
find_subfolders = convert_namespace.s
save_dir_name = convert_namespace.save_dir_name
if convert_namespace.i:
    logging.getLogger().setLevel(logging.INFO)


def convert_run(snapshot_folder, output_folder):
    snapshot_paths = np.asarray(sorted(snapshot_folder.glob(csv_pattern)))  # Find all snapshots

    if snapshot_paths.size == 0:
        warn(f"No snapshot to convert found in {snapshot_folder}")
    else:
        if not output_folder.exists():
            output_folder.mkdir(exist_ok=False)
            info(f"Vtu converted save directory created in {output_folder.resolve()}")
        else:
            raise Exception(f"Save dir already exists: {save_dir.resolve()}. Abort")

        info(f"Output directory {output_folder} was created")
        start = time.time()

        data = pd.read_csv(snapshot_paths[0], delimiter=",")
        data.rename(columns={"r": "a"}, inplace=True)
        column_names = data.columns.values
        vector_names = np.unique(
            [
                c[:-1] for c in column_names
                if (
                    c[:-1] != "" and
                    c[:-1] + "x" in column_names and
                    c[:-1] + "y" in column_names and
                    c[:-1] + "z" in column_names
            )
            ]
        )
        scalar_names = np.array(
            [
                c for c in column_names
                if (
                    c[:-1] not in vector_names and
                    c != "x" and
                    c != "y" and
                    c != "z"
            )
            ]
        )

        for snapshot_path in snapshot_paths:
            start_snap = time.time()
            data = pd.read_csv(snapshot_path, delimiter=",")
            data.rename(columns={"r": "a"}, inplace=True)
            scalars = {sc: data[sc].values for sc in scalar_names}
            vectors = {vc: (data[vc + "x"].values, data[vc + "y"].values, data[vc + "z"].values) for vc in vector_names}
            d = {**scalars, **vectors}
            i = str(snapshot_path).split(".")[-1]
            save_path = output_folder / f"snapshot_{snapshot_folder.parents[1].stem}_{snapshot_folder.parents[0].stem}_{snapshot_folder.stem}_{i}"
            saved_to = pointsToVTK(str(save_path), data["x"].values, data["y"].values, data["z"].values, data=d)
            info(f"Converted snapshot {snapshot_path} to {saved_to} in {time.time() - start_snap}s")

        diff = time.time() - start
        info(f"Converted {snapshot_paths.size} snapshots in {diff:.2f}s ({snapshot_paths.size / diff:.2f} snapshots/s)")

#if len(sys.argv) > 1:
#    raise Exception("Wrong arguments given. Use --help to find list of arguments")

if snapshot_dir:
    user_given_snapshot_directory = snapshot_dir
    snapshot_folder = Path(user_given_snapshot_directory)

    if not snapshot_folder.exists():
        raise Exception(f"Snapshot directory {snapshot_folder.resolve()} does not exist. Please give absolute path.")

    try:
        if find_subfolders:
            sub_folders = sorted(snapshot_folder.glob(run_folder_pattern))

            if len(sub_folders) == 0:
                warn(f"No matching subfolders with pattern {run_folder_pattern} found in {snapshot_folder.resolve()}")

            for sub_folder in sub_folders:
                save_dir = sub_folder / save_dir_name
                convert_run(sub_folder, save_dir)
        else:
            save_dir = snapshot_folder / save_dir_name
            convert_run(snapshot_folder, save_dir)

    except Exception as e:
        warn(f"Could not convert files in {snapshot_folder}:{e}")

else:
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
                                convert_run(run_folder, output_folder)
                            except Exception as e:
                                warn(f"Could not convert files in {run_folder}:{e}")