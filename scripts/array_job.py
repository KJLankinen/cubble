import json
import sys
import os
import copy
import datetime
import subprocess

def create_folder_and_data_file(dir_name, outfile_name, data, inbound):
    print("Creating directory \"" + dir_name + "\".")
    os.makedirs(dir_name)
    data.update((key, val) for (key, val) in inbound.items() if key in data.keys())

    print("Writing input arguments to json \"" + outfile_name + "\".")
    with open(outfile_name, 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)

def main():
    root_dir  = os.path.join(os.environ['WRKDIR'], "cubble")
    default_input_file = os.path.join(root_dir, "input_parameters.json")
    array_param_file = os.path.join(root_dir, "array_parameters.json")
    data_dir = os.path.join(root_dir, "data", datetime.datetime.now().strftime("%d_%m_%Y"))

    build_script = "\
#!/bin/sh\n\
#SBATCH --job-name=cubble_compile\n\
#SBATCH --mem=1G\n\
#SBATCH --time=01:00:00\n\
#SBATCH --gres=gpu:1\n\
#SBATCH --constraint=pascal\n\
#SBATCH --mail-user=juhana.lankinen@aalto.fi\n\
#SBATCH --mail-type=ALL\n\
module purge\n\
module load goolfc/triton-2017a\n\
srun make final -C " + root_dir + " BIN_PATH=/tmp/$SLURM_JOB_ID\n\
cp /tmp/$SLURM_JOB_ID/cubble " + data_dir + "\n\
"

    if not os.path.isdir(root_dir):
        print("Root dir \"" + root_dir + "\" is not a directory.")
        return 1
    
    if not os.path.isfile(default_input_file):
        print("\"" + default_input_file + "\" is not a file.")
        return 1

    if not os.path.isfile(array_param_file):
        print("\"" + array_param_file + "\" is not a file.")
        return 1

    print("Using " + root_dir + " as root dir.")
    print("Using " + default_input_file + " as the default input file.")
    print("Using " + array_param_file + " as the file to modify the default input file with.")

    print("Reading default input arguments.")
    with open(default_input_file) as json_file_handle:
        json_data = json.load(json_file_handle)

    num_runs = 0

    with open(array_param_file) as parameter_file_handle:
        for counter, line in enumerate(parameter_file_handle):
            run_dir = os.path.join(data_dir, "run_" + str(counter))
            outfile_path = os.path.join(run_dir, os.path.split(default_input_file)[1])

            create_folder_and_data_file(run_dir,
                outfile_path,
                copy.deepcopy(json_data),
                json.loads(line.strip()))
            
            num_runs = counter
    
    build_process = subprocess.Popen(["sbatch"], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    build_stdout = build_process.communicate(input=build_script)[0]
    
    print("Output of popen subprocess:")
    print(build_stdout)

if __name__ == "__main__":
    main()


'''
    Array script:
    1. mkdir /tmp/$SLURM_JOB_ID
    2. cd to above
    3. run $WRKDIR/cubble/data/dd_mm_yyyy/cubble with $WRKDIR/cubble/data/dd_mm_yyyy/$SLURM_ARRAY_ID/input_parameters.json
    4. copy files: cp . $WRKDIR/cubble/data/dd_mm_yyyy/$SLURM_ARRAY_ID/
'''
