import json
import sys
import os
import copy

def create_folder_and_data_file(dir_name, outfile_name, data, inbound):
    os.makedirs(dir_name)
    data.update((key, val) for (key, val) in inbound.items() if key in data.keys())

    with open(outfile_name, 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)

def main():

    if len(sys.argv) < 4:
        print("Not enough input arguments. Provide")
        print("\tRoot dir for multiple runs")
        print("\tOriginal input file")
        print("\tFile with arguments for different runs")

        return 1

    root_dir = str(sys.argv[1])
    if not os.path.isdir(root_dir):
        print("Given root dir \"" + root_dir + "\" is not a directory.")
        return 1
    
    original_json = str(sys.argv[2])
    if not os.path.isfile(original_json):
        print("Given json file \"" + original_json + "\" is not a file.")
        return 1

    param_file = str(sys.argv[3])
    if not os.path.isfile(param_file):
        print("Given parameter file \"" + param_file + "\" is not a file.")
        return 1

    print("Using " + root_dir + " as root dir.")
    print("Using " + original_json + " as the original input file.")
    print("Using " + param_file + " as the file to modify the input file with.")

    with open(original_json) as json_file_handle:
        json_data = json.load(json_file_handle)

    num_runs = 0

    with open(param_file) as parameter_file_handle:
        for counter, line in enumerate(parameter_file_handle):
            dir_path = os.path.join(root_dir, "run_" + str(counter), "data")
            outfile_path = os.path.join(dir_path, os.path.split(original_json)[1])

            create_folder_and_data_file(dir_path,
                outfile_path,
                copy.deepcopy(json_data),
                json.loads(line.strip()))
            
            num_runs = counter
    
    print(str(num_runs))

if __name__ == "__main__":
    main()