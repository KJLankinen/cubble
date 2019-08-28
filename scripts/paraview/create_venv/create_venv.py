"""This script creates a virtual environment for the vtu conversion or does nothing if already exists"""
import subprocess
from pathlib import Path
import logging
import os

dir_of_current_file = Path(__file__).parent

###########################################################

shell_script_name = "set_up_venv.sh"  # Shell script that will create the virtual environment
virtual_environment_name = "cubble_python_venv"  # Name of the virtual environment
virtual_environment = dir_of_current_file.parent / virtual_environment_name  # Path to virtual environment
requirements_path = dir_of_current_file / "requirements.txt"  # File that contains the required packages to install
logging.basicConfig(level=logging.INFO)  # Level of output use INFO for basic information and WARN for only warnings

############################################################

shell_script = dir_of_current_file / shell_script_name  # Path to shell script


def prompt_bool():
    """
    Function that waits for user to enter "y" or "n"

    Returns:
        str: User input; "y" or "n"

    """
    x = input()
    while x != "y" and x != "n":
        print("Please enter y or n")
        x = input()
    return x


# Check if virtual environment already exists. Only create, if does not
if not virtual_environment.exists():
    print(
        f'No virtual python environment called "{virtual_environment_name}" found for the project\n\n'
        'In order to use the full functionality it is recommended to use the required modules in a virtual environment.\n'
        'This only needs to be done once for every local repository and will set up an environment for using python '
        'packages without needing to install them globally\n\n'
        'Would you like to create one now? [y/n]'
    )

    x = prompt_bool()

    if x == "y":
        print('Creating new virtual environment')
        try:
            os.chmod(shell_script, 0o740)  # Set the shell script as excecutable
            # Excecute shell script and create new virtual environment. Installs the required packages
            subprocess.run(["sh",  f"{shell_script.resolve()}", f"{virtual_environment.resolve()}",
                            f"{requirements_path.resolve()}"], check=True)
            logging.info(f'New virtual environment in {virtual_environment}')
        except subprocess.CalledProcessError as e:
            logging.warning(e.output)
        finally:
            os.chmod(shell_script, 0o640)
    elif x == "n":
        # Possiblity of ignoring this procedure by creating empty file
        print(f"You can silence this message in the future. This will create an empty directory in "
              f"{virtual_environment.resolve()}\n"
              f"Ignore future prompt to create new environment? [y/n]")

        x = prompt_bool()
        if x == "y":
            virtual_environment.mkdir()
            print(f"Created empty folder {virtual_environment.resolve()}. "
                  f"Future prompts will be silenced until the directory is removed.")

        print("Continuing without creation of environment ...")

else:
    logging.info(f'No action required: Python virtual environment is ready set up in {virtual_environment.resolve()}')

