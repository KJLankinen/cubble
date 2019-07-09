import json
import sys
import os
import copy
import datetime
import subprocess
import pwd

def create_folder_and_data_file(dir_name, outfile_name, data, inbound):
    os.makedirs(dir_name)
    
    for key, val in inbound.items():
	if key in data.keys():
	    data.update({key:val})
	else:
	    print("Key not found in default input file: " + key)

    with open(outfile_name, 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)

def main():

    if len(sys.argv) < 2:
        print("Give a (descriptive) name for the sub directory the simulation data is saved to.")
        return 1
    
    sub_dir = sys.argv[1]
    root_dir  = os.path.join(os.environ['WRKDIR'], "cubble")
    make_dir = os.path.join(root_dir, "final")
    default_input_file = os.path.join(root_dir, "input_parameters.json")
    array_param_file = os.path.join(root_dir, "array_parameters.json")
    data_dir = os.path.join(root_dir, "data", datetime.datetime.now().strftime("%d_%m_%Y"), sub_dir)
    executable_path = os.path.join(data_dir, "cubble")

    compile_script = "\
#!/bin/bash\n\
#SBATCH --job-name=cubble_compile\n\
#SBATCH --mem=100M\n\
#SBATCH --time=00:10:00\n\
#SBATCH --gres=gpu:1\n\
#SBATCH --constraint='volta'\n\
#SBATCH --mail-user=juhana.lankinen@aalto.fi\n\
#SBATCH --mail-type=ALL\n\
module purge\n\
module load cuda/10.0.130 gcc/6.3.0\n\
mkdir /tmp/$SLURM_JOB_ID\n\
srun make -C " + make_dir + " BIN_PATH=/tmp/$SLURM_JOB_ID\n\
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

    if not os.path.isdir(make_dir):
	print("Make dir \"" + make_dir + "\" is not a directory.")
	return 1

    print("Using " + root_dir + " as root dir.")
    print("Using " + make_dir + " as the makefile directory.")
    print("Using " + default_input_file + " as the default input file.")
    print("Using " + array_param_file + " as the file to modify the default input file with.")
    print("Using " + data_dir + " as the data directory for this simulation run.")
    
    print("Launching process for compiling the binary.")
    compile_process = subprocess.Popen(["sbatch"], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    compile_stdout = compile_process.communicate(input=compile_script)[0]
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

            create_folder_and_data_file(run_dir,
                outfile_path,
                copy.deepcopy(json_data),
                json.loads(line.strip()))
            
            num_runs = counter

    array_data_dir = os.path.join(data_dir, "run_$SLURM_ARRAY_TASK_ID")
    array_input_path = os.path.join(array_data_dir, os.path.split(default_input_file)[1])
    array_temp_dir = "/tmp/$SLURM_JOB_ID"

    array_script = "\
#!/bin/bash\n\
#SBATCH --job-name=cubble\n\
#SBATCH --mem=4G\n\
#SBATCH --time=120:00:00\n\
#SBATCH --gres=gpu:1\n\
#SBATCH --constraint=\"volta\"\n\
#SBATCH --mail-user=juhana.lankinen@aalto.fi\n\
#SBATCH --mail-type=ALL\n\
#SBATCH --dependency=aftercorr:" + compile_slurm_id + "\n\
#SBATCH --array=0-" + str(num_runs) + "\n\
#SBATCH --signal=USR1@180
module purge\n\
module load cuda/10.0.130 gcc/6.3.0\n\
mkdir " + array_temp_dir + "\n\
cd " + array_temp_dir + "\n\
srun " + executable_path + " " + array_input_path + " output_parameters.json\n\
mv -f " + array_temp_dir + "/* " + array_data_dir + "\n\
"

    print("Launching an array of processes that run the simulation.")
    array_process = subprocess.Popen(["sbatch"], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    array_stdout = array_process.communicate(input=array_script)[0]

    if array_process.returncode != 0:
        print("Array process submission was not successful!")
        return array_process.returncode

    current_user = pwd.getpwuid(os.getuid()).pw_name

    squeue_process = subprocess.Popen(["squeue", "-u", current_user], stdout=subprocess.PIPE)
    print("Slurm queue of the current user:")
    print(squeue_process.communicate()[0])
    print("\nJob submission done!")


if __name__ == "__main__":
    main()
