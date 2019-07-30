from pyevtk.vtk import VtkGroup
from pyevtk.hl import pointsToVTK
import logging
from pathlib import Path
import sys
import numpy as np

########################################################################################################################
# User adjustable variables
########################################################################################################################

csv_pattern = 'snapshot.csv.[0-9]*'
output_folder_name = "vtu_snapshots"
run_folder_pattern = 'run_[0-9]*'
logging.basicConfig(level=logging.INFO)  # logging.INFO or logging.DEBUG or logging.WARNING

########################################################################################################################
directory_of_this_script = Path(__file__).parent  # Current script file folder path
data_directory = directory_of_this_script.parent / "data"
user_gave_arguments = False
info = logging.info; debug = logging.debug; warn = logging.warning


def convert_run(snapshot_folder, output_folder):
    snapshot_paths = np.asarray(sorted(snapshot_folder.glob(csv_pattern)))  # Find all snapshots

    if snapshot_paths.size == 0:
        warn(f"No snapshot to convert found in {snapshot_folder}")
    else:
        output_folder.mkdir(exist_ok=False)
        info(f"Output directory {output_folder} was created")
        for snapshot_path in snapshot_paths:
            data = np.loadtxt(snapshot_path, delimiter=",", skiprows=1, unpack=True)
            x, y, z, r, vx, vy, vz, path, dist = [col.flatten() for col in np.split(data.T, 9, 1)]
            i = str(snapshot_path).split(".")[-1]
            save_path = output_folder / f"snapshot_{i}"
            saved_to = pointsToVTK(str(save_path), x, y, z, data={"radius": r, "velocity": (vx, vy, vz), "path": path, "dist": dist})
            info(f"Converted snapshot {snapshot_path} to {saved_to}")

if len(sys.argv) == 2:
    raise Exception("Too few arguments given. Expected two: location of csv snapshots and folder name to save to")

elif len(sys.argv) == 3:
    user_given_snapshot_directory = sys.argv[1]
    user_given_save_directory_name = sys.argv[2]
    snapshot_folder = Path(user_given_snapshot_directory)
    user_gave_arguments = True
    if not snapshot_folder.exists():
        raise Exception("Given snapshot directory does not exist. Stop")
    run_folder_paths = np.asarray(sorted(snapshot_folder.glob(run_folder_pattern)))

    for run_folder in run_folder_paths:
        try:
            convert_run(run_folder, run_folder / user_given_save_directory_name)
        except:
            print(f"Could not convert files in {run_folder}")

elif len(sys.argv) > 3:
    raise Exception("Too many arguments given. Expected only two: location of csv snapshots and folder name to save to")

else:
    warn("No arguments given, assuming that everything in data folder should be converted")

    print(f"\nAre you sure you want to find all possible snapshots in {data_directory.resolve()} and convert them "
          f"\n(this is NOT recommended and may take a long time)? [y/n]:")
    x = input()
    while x != "y" and x != "n":
        print("Please enter y or n")
        x = input()
    return x

    if x == "y":

        sub_folders = sorted(data_directory.glob("[0-9][0-9]_[0-9][0-9]_[0-9][0-9][0-9][0-9]"))

        for sub_folder in sub_folders:
            sub_sub_folders = sorted(sub_folder.glob("*"))
            for sub_sub_folder in sub_sub_folders:
                if sub_folder.is_dir():
                    run_folder_paths = np.asarray(sorted(sub_sub_folder.glob(run_folder_pattern)))
                    for run_folder in run_folder_paths:
                        output_folder = run_folder / output_folder_name
                        if not output_folder.exists():
                            try:
                                convert_run(run_folder, output_folder)
                            except:
                                print(f"Could not convert files in {run_folder}")