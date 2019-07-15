#!/usr/bin/python

import json
import sys
import os
import copy
import datetime
import subprocess
import pwd
import shutil

def create_folder_and_data_file(dir_name, outfile_name, data, inbound):
    os.makedirs(dir_name)

    for key, val in inbound.items():
	if key in data.keys():
	    data.update({key:val})
	else:
	    print("Key not found in default input file: " + key)

    with open(outfile_name, 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)

class File:
    name = ""
    path = ""
    stem = ""

    def __init__(self, name, root, stem=None):
        self.name = name
        if stem is not None:
            self.stem = stem
            self.path = os.path.join(root, stem, name)
        else:
            self.path = os.path.join(root, name)

def main():
    if len(sys.argv) < 2:
        print("Give a (descriptive) name for the sub directory the simulation data is saved to.")
        return 1
    
    sb_modules =    "cuda/10.0.130 gcc/6.3.0"
    sb_mem =        "32G"
    sb_time =       "120:00:00"
    sb_gres =       "gpu:1"
    sb_constraint = "\"volta\""
    sb_mail_user =  "juhana.lankinen@aalto.fi"
    sb_mail_type =  "ALL"
    sb_signal =     "USR1@180"
    
    root_dir =              File("cubble", os.environ['WRKDIR'])
    data_dir =              File(sys.argv[1], root_dir.path, os.path.join("data", datetime.datetime.now().strftime("%d_%m_%Y")))
    array_work_dir =        File("run_$RUN_NUM", root_dir.path)
    make_file =             File("makefile", root_dir.path, "final")
    default_input =         File("input_parameters.json", root_dir.path)
    arr_params =            File("array_parameters.json", root_dir.path)
    executable =            File("cubble", root_dir.path, data_dir.path)
    array_work_dir =        File("run_$RUN_NUM", root_dir.path, data_dir)
    array_input =           File(os.path.split(default_input_file)[1], array_work_dir.path)
    continue_script =       File("continue_script.sh", array_work_dir.path)
    binary =                File("state.bin", array_work_dir.path) 
    result_file =           File("results.dat", array_work_dir.path)
    temp_dir =              File("$TEMP_DIR", "/tmp")
    
    print("Copying makefile from " + make_file.path + " to " + data_dir.path + "/" + make_file.name)
    os.makedirs(data_dir.path)
    shutil.copyfile(make_file.path, os.path.join(data_dir.path, make_file.name))
    make_file = File(make_file.name, data_dir.path)

    compile_script_str = "\
#!/bin/bash\n\
#SBATCH --job-name=cubble_compile\n\
#SBATCH --mem=100M\n\
#SBATCH --time=00:10:00\n\
#SBATCH --gres=" + sb_gres + "\n\
#SBATCH --constraint=" + sb_constraint + "\n\
#SBATCH --mail-user=" + sb_mail_user + "\n\
#SBATCH --mail-type=" + sb_mail_type + "\n\
TEMP_DIR=$SLURM_JOB_ID\n\
module load " + sb_modules + "\n\
mkdir " + temp_dir.path + "\n\
srun make -C " + make_file.path + " BIN_PATH=" + temp_dir.path + "\n\
cp " + temp_dir.path + "/" + executable_name.name + " " + data_dir.path + "\
"
    if not os.path.isdir(root_dir.path):
        print("Root dir \"" + root_dir.path + "\" is not a directory.")
        return 1

    if not os.path.isfile(default_input.path):
        print("\"" + default_input.path + "\" is not a file.")
        return 1

    if not os.path.isfile(arr_params):
        print("\"" + arr_params.path + "\" is not a file.")
        return 1

    if not os.path.isfile(make_file.path):
        print("Makefile \"" + make_file.path + "\" is not a file.")
        return 1

    print("Using " + root_dir.path + " as root dir.")
    print("Using " + data_dir.path + " as the data directory for this simulation run.\n")
    print("Using " + make_file.path + " as the makefile.")
    print("Using " + default_input.path + " as the default input file.")
    print("Using " + arr_params.path + " as the file to modify the default input file with.")

    print("Launching process for compiling the binary.")
    compile_process = subprocess.Popen(["sbatch"], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    compile_stdout = compile_process.communicate(input=compile_script_str)[0]
    compile_slurm_id = str(compile_stdout).strip().split(" ")[-1]

    if compile_process.returncode != 0:
        print("Compile process submission was not successful!")
        return compile_process.returncode

    print("Reading default input arguments.")
    with open(default_input_file) as json_file_handle:
        json_data = json.load(json_file_handle)

    num_runs = 0
    print("Creating directories and input files.")
    with open(array_param_file) as parameter_file_handle:
        for counter, line in enumerate(parameter_file_handle):
            run_dir = os.path.join(data_dir, "run_" + str(counter))
            outfile_path = os.path.join(run_dir, os.path.split(default_input_file)[1])
            create_folder_and_data_file(run_dir, outfile_path, copy.deepcopy(json_data), json.loads(line.strip()))
            num_runs = counter

    continue_script_str = "\
#!/bin/bash\n\
#SBATCH --job-name=cubble\n\
#SBATCH --mem=" + sb_mem + "\n\
#SBATCH --time=" + sb_time + "\n\
#SBATCH --gres=" + sb_gres + "\n\
#SBATCH --constraint=" + sb_constraint + "\n\
#SBATCH --mail-user=" + sb_mail_user + "\n\
#SBATCH --mail-type=" + sb_mail_type + "\n\
#SBATCH --signal=" + sb_signal + "\n\
RUN_NUM=$1\n\
TIMES_CALLED=$2\n\
TEMP_DIR=$SLURM_JOB_ID\n\
module load " + sb_modules + "\n\
mkdir " + temp_dir.path + "\n\
cd " + temp_dir.path + "\n\
if [ -f " + result_file.path + " ]; then cp " + result_file.path + " .; fi\n\
srun " + executable.path + " " + binary.path + " " + binary.name + "\n\
rm " + binary.path + "\n\
mv -f " + temp_dir.path + "/* " + array_work_dir.path + "\n\
cd " + array_work_dir.path + "\n\
if [ -f " + binary.name + " ] && [ -f " + continue_script.name + " ] && [[ ( $TIMES_CALLED < 3 ) ]]; \
then cd " + root_dir.path + "; sbatch " + continue_script.path + " $RUN_NUM $(($TIMES_CALLED + 1)); \
elif [ -f " + continue_script.name + " ]; then rm " + continue_script.name + "; fi\
"
    # Important to echo the continue script to file with single quotes to avoid bash variable expansion
    # See the second to last line of this script.
    array_script_str = "\
#!/bin/bash\n\
#SBATCH --job-name=cubble\n\
#SBATCH --mem=" + sb_mem + "\n\
#SBATCH --time=" + sb_time + "\n\
#SBATCH --gres=" + sb_gres + "\n\
#SBATCH --constraint=" + sb_constraint + "\n\
#SBATCH --mail-user=" + sb_mail_user + "\n\
#SBATCH --mail-type=" + sb_mail_type + "\n\
#SBATCH --dependency=aftercorr:" + compile_slurm_id + "\n\
#SBATCH --array=0-" + str(num_runs) + "\n\
#SBATCH --signal=" + sb_signal + "\n\
RUN_NUM=$SLURM_ARRAY_TASK_ID\n\
TEMP_DIR=$SLURM_JOB_ID\n\
module load " + sb_modules + "\n\
mkdir " + temp_dir.path + "\n\
cd " + temp_dir.path + "\n\
srun " + executable.path + " " + array_input.path + " " + binary.name + "\n\
mv -f " + temp_dir.path + "/* " + array_work_dir.path + "\n\
cd " + array_work_dir.path + "\n\
if [ -f " + binary.name + " ]; then echo \'" + continue_script_str + "\' > " + continue_script.name + "; fi\n\
if [ -f " + continue_script.name + " ]; then cd " + root_dir.path + "; sbatch " + continue_script.path + " $RUN_NUM 1; fi\
"

    print("Launching an array of processes that run the simulation.\n")
    array_process = subprocess.Popen(["sbatch"], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    array_stdout = array_process.communicate(input=array_script_str)[0]

    if array_process.returncode != 0:
        print("Array process submission was not successful!")
        return array_process.returncode

    squeue_process = subprocess.Popen(["slurm", "q"], stdout=subprocess.PIPE)
    print("Slurm queue:")
    print(squeue_process.communicate()[0])
    print("\nJob submission done!")

if __name__ == "__main__":
    main()
