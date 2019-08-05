from pyevtk.vtk import VtkGroup
from pyevtk.hl import pointsToVTK
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import time

########################################################################################################################
# User adjustable variables
########################################################################################################################

csv_pattern = 'snapshot.csv.[0-9]*'
default_output_folder_name = "vtu_snapshots"
run_folder_pattern = 'run_[0-9]*'
logging.basicConfig(level=logging.INFO)  # logging.INFO or logging.DEBUG or logging.WARNING

########################################################################################################################
directory_of_this_script = Path(__file__).parent  # Current script file folder path
data_directory = directory_of_this_script / "data"
info = logging.info; debug = logging.debug; warn = logging.warning


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


if len(sys.argv) > 1:
    user_given_snapshot_directory = sys.argv[1]
    snapshot_folder = Path(user_given_snapshot_directory)
    save_dir = None
    if not snapshot_folder.exists():
        raise Exception(f"Snapshot directory {snapshot_folder.resolve()} does not exist. Please give absolute path.")

    if len(sys.argv) == 2:
        save_dir = snapshot_folder / default_output_folder_name
    elif len(sys.argv) == 3:
        user_given_save_directory = sys.argv[2]
        save_dir = Path(user_given_save_directory)

    elif len(sys.argv) > 3:
        raise Exception("Too many arguments given. Expected one or two: location of csv snapshots "
                        "(and optionally folder path to save to relative to snapshot dir, otherwise inside snapshot dir)")
    try:
        convert_run(snapshot_folder, save_dir)
    except Exception as e:
        warn(f"Could not convert files in {snapshot_folder}:{e}")

else:
    print("No arguments given, assuming that everything in data folder should be converted")

    print(f"\nAre you sure you want to find all possible snapshots in {data_directory.resolve()} and convert them "
          f"\n(this is NOT recommended and may take a long time)? [y/n]:")
    x = input()
    while x != "y" and x != "n":
        print("Please enter y or n")
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
                                print(f"Could not convert files in {run_folder}:{e}")