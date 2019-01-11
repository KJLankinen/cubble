import json
import sys
import os
import copy
import datetime

def create_folder_and_data_file(dir_name, outfile_name, data, inbound):
    os.makedirs(dir_name)
    data.update((key, val) for (key, val) in inbound.items() if key in data.keys())

    with open(outfile_name, 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)

def main():
    ROOT_DIR  = os.path.join(os.environ['WRKDIR'], "cubble")
    DEFAULT_INPUT_FILE = os.path.join(ROOT_DIR, "input_parameters.json")
    ARRAY_PARAM_FILE = os.path.join(ROOT_DIR, "array_parameters.json")
    
    date_str = datetime.datetime.now().strftime("%d_%m_%Y")

    if not os.path.isdir(ROOT_DIR):
        print("Root dir \"" + ROOT_DIR + "\" is not a directory.")
        return 1
    
    if not os.path.isfile(DEFAULT_INPUT_FILE):
        print("\"" + DEFAULT_INPUT_FILE + "\" is not a file.")
        return 1

    if not os.path.isfile(ARRAY_PARAM_FILE):
        print("\"" + ARRAY_PARAM_FILE + "\" is not a file.")
        return 1

    print("Using " + ROOT_DIR + " as root dir.")
    print("Using " + DEFAULT_INPUT_FILE + " as the default input file.")
    print("Using " + ARRAY_PARAM_FILE + " as the file to modify the default input file with.")

    with open(DEFAULT_INPUT_FILE) as json_file_handle:
        json_data = json.load(json_file_handle)

    num_runs = 0

    with open(ARRAY_PARAM_FILE) as parameter_file_handle:
        for counter, line in enumerate(parameter_file_handle):
            dir_path = os.path.join(ROOT_DIR, date_str, "data", "run_" + str(counter))
            outfile_path = os.path.join(dir_path, os.path.split(DEFAULT_INPUT_FILE)[1])

            create_folder_and_data_file(dir_path,
                outfile_path,
                copy.deepcopy(json_data),
                json.loads(line.strip()))
            
            num_runs = counter
    
    print(str(num_runs))

if __name__ == "__main__":
    main()


'''
    Build script:
    1. CD to $WRKDIR
    2. Make
    3. mkdir under $WRKDIR/cubble/data/dd_mm_yyyy for each line in multirun json
    4. copy built bin/cubble to dd_mm_yyyy
    5. process multirun json and add input json to each child dir
    6. make clean

    Array script:
    1. mkdir /tmp/$SLURM_JOB_ID
    2. cd to above
    3. run $WRKDIR/cubble/data/dd_mm_yyyy/cubble with $WRKDIR/cubble/data/dd_mm_yyyy/$SLURM_ARRAY_ID/input_parameters.json
    4. copy files: cp . $WRKDIR/cubble/data/dd_mm_yyyy/$SLURM_ARRAY_ID/
'''
