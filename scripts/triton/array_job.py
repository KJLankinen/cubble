#!/usr/bin/python

import json
import sys
import os
import copy
import datetime
import subprocess
import shutil
import errno

def main():
    if len(sys.argv) < 2:
        print("Give a (descriptive) name for the sub directory the simulation data is saved to.")
        return 1

    sb_name =           sys.argv[1] 
    sb_compile_name =   "cubble_compile_" + sys.argv[1]
    sb_modules =        "cuda/10.0.130 gcc/6.3.0"
    sb_mem =            "32G"
    sb_time =           "120:00:00"
    sb_gres =           "gpu:1"
    sb_constraint =     "\"volta\""
    sb_mail_user =      os.popen('git config user.email').read().replace("\n", "")
    sb_mail_type =      "ALL"

    print("\nUsing the following paths & files:")
    print("----------------------------------\n")
    root_dir =              File("cubble", os.environ['WRKDIR'], None, False, True)  ###
    src_dir =               File("src", root_dir.path)
    incl_dir =              File("incl", root_dir.path)
    data_dir =              File(sb_name,
                                 root_dir.path,
                                 os.path.join("data", datetime.datetime.now().strftime("%d_%m_%Y")),
                                 True)
    make_file =             File("makefile", root_dir.path, "final", False, True)
    default_input =         File("input_parameters.json", root_dir.path, None, False, True)
    arr_params =            File("array_parameters.json", root_dir.path, None, False, True)
    executable =            File("cubble", data_dir.path)
    array_work_dir =        File("run_$RUN_NUM", data_dir.path)
    array_input =           File(default_input.name, array_work_dir.path)
    result_file =           File("results.dat", array_work_dir.path)
    temp_dir =              File("$TEMP_DIR", "/tmp")
    print("----------------------------------\n")
    
    print("Copying makefile from " + make_file.path + " to " + data_dir.path + "/" + make_file.name)
    shutil.copyfile(make_file.path, os.path.join(data_dir.path, make_file.name))
	
    compile_script_str = "\
#!/bin/bash\n\
#SBATCH --job-name=" + sb_compile_name + "\n\
#SBATCH --mem=1G\n\
#SBATCH --time=00:10:00\n\
#SBATCH --gres=" + sb_gres + "\n\
#SBATCH --constraint=" + sb_constraint + "\n\
#SBATCH --mail-user=" + sb_mail_user + "\n\
#SBATCH --mail-type=" + sb_mail_type + "\n\
TEMP_DIR=$SLURM_JOB_ID\n\
module load " + sb_modules + "\n\
mkdir " + temp_dir.path + "\n\
srun make -C " + data_dir.path + " SRC_PATH=" + src_dir.path + " BIN_PATH=" + temp_dir.path + " INCL=-I" + incl_dir.path + "\n\
cp " + temp_dir.path + "/" + executable.name + " " + data_dir.path
    
    print("Launching process for compiling the binary.")
    compile_process = subprocess.Popen(["sbatch"], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    compile_stdout = compile_process.communicate(input=compile_script_str.encode())[0]
    compile_slurm_id = str(compile_stdout.decode()).strip().split(" ")[-1]

    if compile_process.returncode != 0:
        print("Compile process submission was not successful!")
        return compile_process.returncode
    else:
        print(str(compile_stdout.decode()))

    print("Reading default input arguments.")
    with open(default_input.path) as json_file_handle:
        json_data = json.load(json_file_handle)

    num_runs = 0
    print("Creating directories and input files.")
    with open(arr_params.path) as parameter_file_handle:
        for counter, line in enumerate(parameter_file_handle):
            run_dir = os.path.join(data_dir.path, "run_" + str(counter))
            outfile_path = os.path.join(run_dir, os.path.split(default_input.path)[1])
            create_folder_and_data_file(run_dir, outfile_path, copy.deepcopy(json_data), json.loads(line.strip()))
            num_runs = counter

    array_script_str = "\
#!/bin/bash\n\
#SBATCH --job-name=cubble_" + sb_name + "\n\
#SBATCH --mem=" + sb_mem + "\n\
#SBATCH --time=" + sb_time + "\n\
#SBATCH --gres=" + sb_gres + "\n\
#SBATCH --constraint=" + sb_constraint + "\n\
#SBATCH --mail-user=" + sb_mail_user + "\n\
#SBATCH --mail-type=" + sb_mail_type + "\n\
#SBATCH --dependency=aftercorr:" + compile_slurm_id + "\n\
#SBATCH --array=0-" + str(num_runs) + "\n\
RUN_NUM=$SLURM_ARRAY_TASK_ID\n\
TEMP_DIR=$SLURM_JOB_ID\n\
module load " + sb_modules + "\n\
mkdir " + temp_dir.path + "\n\
cd " + temp_dir.path + "\n\
srun " + executable.path + " " + array_input.path + "\n\
tar czf snapshots.tar.gz snapshot.csv.*\n\
rm snapshot.csv.*\n\
mv -f " + temp_dir.path + "/* " + array_work_dir.path + "\n\
cd " + array_work_dir.path + "\n"

    print("Launching an array of processes that run the simulation.")
    array_process = subprocess.Popen(["sbatch"], stdout=subprocess.PIPE, stdin=subprocess.PIPE)
    array_stdout = array_process.communicate(input=array_script_str.encode())[0]

    if array_process.returncode != 0:
        print("Array process submission was not successful!")
        return array_process.returncode
    else:
        print(str(array_stdout.decode()))

    squeue_process = subprocess.Popen(["slurm", "q"], stdout=subprocess.PIPE)
    print("Slurm queue:")
    print(str(squeue_process.communicate()[0].decode()))
    print("\nJob submission done!")

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

    def __init__(self, name, root, stem=None, create=False, checkIfExists=False):
        self.name = name
        if stem is not None:
            self.stem = stem
            self.path = os.path.join(root, stem, name)
        else:
            self.path = os.path.join(root, name)

        if checkIfExists and not os.path.isdir(self.path) and not os.path.isfile(self.path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.path)

        if create and not os.path.isdir(self.path) and not os.path.isfile(self.path):
            print("Creating: " + self.path)
            os.makedirs(self.path)
        elif create and (os.path.isdir(self.path) or os.path.isfile(self.path)):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), self.path)
        else:
            print(self.path)




if __name__ == "__main__":
    main()
