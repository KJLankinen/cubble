"""
This script creates a Paraview Python save state file for loading bubble snapshots

A template save state file is used which it duplicates while the line that is responsible for telling Paraview which
files to load is replaced by locations of the snapshots. For this the snapshots are first found in a given directory

:param (first argument): The directory where the snapshots are located. If left empty will use directory "data"
"""
import fileinput
import logging
from pathlib import Path
import shutil
import sys
import numpy as np

directory_of_this_script = Path(__file__).parent  # Current script file folder path
########################################################################################################################
# User adjustable variables
########################################################################################################################

data_folder = directory_of_this_script.parent / "data"
python_template_name = "paraview_snapshot.python_template"
final_save_state_file = "paraview_snapshot_save_state.py"
csv_pattern = 'snapshot.csv.[0-9]*'
run_folder_pattern = 'run_[0-9]*'
line_in_python_template_to_be_replaced = "# CSV files paths"
logging.basicConfig(level=logging.INFO)  # logging.INFO or logging.DEBUG

########################################################################################################################
info = logging.info
debug = logging.debug

info("Starting creation of Paraview Python Save State file from snapshots")

if len(sys.argv) == 2:
    user_given_directory = sys.argv[1]
    data_folder = Path(user_given_directory)
elif len(sys.argv) > 2:
    raise Exception("Too many arguments given. Expected only one: directory location of csv snapshots")

if not data_folder.exists():
    raise Exception("Directory given does not exist")


template = directory_of_this_script / python_template_name  # Template path
debug(f"Template located at {str(template)}")


def create_save_state_of_run(run_folder):
    snapshot_folder = data_folder / run_folder
    data_files_paths = np.asarray(sorted(snapshot_folder.glob(csv_pattern)))  # Find all snapshots with the defined pattern

    # Need to first sort the the snapshots according to their number, otherwise loaded in wrong order
    data_files_paths_numbers = [int(str(file).split(".")[-1]) for file in data_files_paths]
    sorted_indices = np.argsort(data_files_paths_numbers)
    data_files_paths = data_files_paths[sorted_indices]

    # Create a long string of all the snapshot paths
    data_files_paths_strings = [str(file.resolve()) for file in data_files_paths]
    string_separation = "',\n    '"  # For convenience and readability in final file
    data_files_paths_strings_joined = f"    '{string_separation.join(data_files_paths_strings)}'"
    debug(f"Following snapshot files were found: {data_files_paths_strings_joined}")

    # Copy the template into a new file that will be modified
    created_state = snapshot_folder / final_save_state_file
    shutil.copy(str(template), str(created_state))
    debug(f"Copied fresh template as {str(created_state)}")

    # Go through every line in the template and replace the line that states which snapshots should be loaded
    for line in fileinput.FileInput(str(created_state), inplace=1):
        line = line.replace(line_in_python_template_to_be_replaced, data_files_paths_strings_joined)

        # Unintuitive: writes the line in the file, i.e. does not print in console. If left blank, file will be empty
        sys.stdout.write(line)

    debug(f"Finished creating save state file located at {str(created_state)} from {data_files_paths.size} snapshots")


run_folder_paths = np.asarray(sorted(data_folder.glob(run_folder_pattern)))
for run_folder in run_folder_paths:
    info(f"Creating save state file in folder {str(run_folder)}")
    create_save_state_of_run(run_folder)

info(f"Finished creating Paraview save state files from {run_folder_paths.size} run(s) located at {str(data_folder)}")

