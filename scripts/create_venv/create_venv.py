import subprocess
from pathlib import Path
import logging
import os

dir_of_current_file = Path(__file__).parent

###########################################################

shell_script_name = "set_up_venv.sh"
virtual_environment_name = "cubble_python_venv"
virtual_environment = dir_of_current_file.parent / virtual_environment_name
requirements_path = dir_of_current_file / "requirements.txt"
logging.basicConfig(level=logging.INFO)


############################################################

shell_script = dir_of_current_file / shell_script_name


def prompt_bool():
    x = input()
    while x != "y" and x != "n":
        print("Please enter y or n")
        x = input()
    return x


if not virtual_environment.exists():
    print(
        f'No virtual python environment called "{virtual_environment_name}" found for the project\n\n'
        'In order to use the full functionality it is recommended to use the required modules in a virtual environment.\n'
        'This only needs to be done once for every local repository and will set up an environment for using python packages without needing to install them globally\n\n'
        'Would you like to create one now? [y/n]'
    )

    x = prompt_bool()

    if x == "y":
        print('Creating new virtual environment')
        try:
            os.chmod(shell_script, 0o740)
            subprocess.run(["sh",  f"{shell_script.resolve()}", f"{virtual_environment.resolve()}", f"{requirements_path.resolve()}"], check=True)
            logging.info(f'New virtual environment in {virtual_environment}')
        except subprocess.CalledProcessError as e:
            logging.warning(e.output)
        finally:
            os.chmod(shell_script, 0o640)
    elif x == "n":
        print(f"You can silence this message in the future. This will create an empty directory in {virtual_environment.resolve()}\n"
              f"Ignore future prompt to create new environment? [y/n]")

        x = prompt_bool()
        if x == "y":
            virtual_environment.mkdir()
            print(f"Created empty folder {virtual_environment.resolve()}. "
                  f"Future prompts will be silenced until the directory is removed.")

        print("Continuing without creation of environment ...")

else:
    logging.info(f'No action required: Python virtual environment is ready set up in {virtual_environment.resolve()}')

