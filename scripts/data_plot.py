import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import sys
import os
import json

def plot_data_loglog(data_file, json_file, ax):
    print("Plotting data from \"" + str(data_file) + "\" using \"" + str(json_file) + "\" for some parameters.")
    data = np.loadtxt(data_file)
    x = data[:, 0]
    y = data[:, 1]
            
    with open(json_file, 'r') as f:
        decoded_json = json.load(f)
            
    phi = decoded_json["PhiTarget"]
    kappa = decoded_json["Kappa"]
    label_str = r"$\phi=$" + str(phi) + r", $\kappa=$" + str(kappa)
    
    ax.loglog(x, y, '+', linewidth=1.5, label=label_str)

def plot_line(ax, alpha, x, y, line_color, label_str):
    ax.loglog(x, y, '--', color=line_color, linewidth=2.0, label=label_str)

def plot_relative_radius(ax, parent_dir, data_file_name, json_file_name, num_plots):
    
    ax.xaxis.label.set_fontsize(20)
    ax.yaxis.label.set_fontsize(20)
    ax.set_xlim(1, 12000)
    ax.set_ylim(0.9, 15)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\frac{R}{\langle R_{in} \rangle}$", rotation=0)
    ax.grid(1)

    child_dirs = [name for name in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, name))]
    json_files = []
    data_files = []
    phis = []
    kappas = []

    # Gather files and sort them according to phi
    for dir in child_dirs:
        for root, dirs, files in os.walk(os.path.join(parent_dir, dir)):
            if not files:
                continue

            if data_file_name in files and json_file_name in files:
                data_file = os.path.join(root, data_file_name)
                json_file = os.path.join(root, json_file_name)
        
                if not os.path.exists(data_file) or not os.path.exists(json_file):
                    print("Path constructed from given arguments does not exist. Path: \"" + str(root) + "\".")
                    print("Data file path: \"" + str(data_file) + "\".")
                    print("Json file path: \"" + str(json_file) + "\".")
                    print("Given arguments:\nparent directory: \"" + str(parent_dir) + "\"\ndata file name: \"" + str(data_file_name) + "\"\njson file name: \"" + str(json_file_name) + "\"\nnumber of files: " + str(num_files) + ".")
                    sys.exit(1)

                with open(json_file, 'r') as f:
                    decoded_json = json.load(f)
            
                phis.append(float(decoded_json["PhiTarget"]))
                kappas.append(float(decoded_json["Kappa"]))
                json_files.append(json_file)
                data_files.append(data_file)

    indices = list(reversed(np.lexsort((np.array(kappas), np.array(phis)))))
    json_files = [json_files[idx] for idx in indices]
    data_files = [data_files[idx] for idx in indices]

    # line
    alpha = 0.48
    x = np.linspace(30, 1250, 1000)
    y = pow(0.3 * x, alpha)
    line_color = (1, 0, 0)
    label_str = r"$k \tau^{\alpha}$" + r", $\alpha=$" + str(alpha)
    plot_line(ax, alpha, x, y, line_color, label_str)

    # Plot
    for i in range(min(len(data_files), max(num_plots, 0))):
        plot_data_loglog(data_files[i], json_files[i], ax)

    # line
    alpha = 0.33
    x = np.linspace(100, 58000, 1000)
    y = pow(0.07 * x, alpha)
    label_str = r"$k \tau^{\alpha}$" + r", $\alpha=$" + str(alpha)
    line_color = (0, 0, 0)
    plot_line(ax, alpha, x, y, line_color, label_str)

    ax.legend(loc='upper left')

    plt.show()
    
def main():
    arguments = sys.argv
    data_file_name = "output.dat"
    json_file_name = "output_parameters.json"
    num_plots = 999
    
    if len(arguments) < 2:
        print("Too few arguments given. Give:\n\tthe parent folder of data files\n\t# of plots\n\tname of data file\n\tname of saved json file\nas arguments.")
        sys.exit(1)
    elif len(arguments) < 3:
        print("Using default names for data and json files, \"" + data_file_name + "\" and \"" + json_file_name + "\" respectively and plotting all data files.")
    elif len(arguments) < 4:
        num_plots = int(arguments[2])
        print("Using default names for data and json files, \"" + data_file_name + "\" and \"" + json_file_name + "\" respectively and plotting " + str(num_plots) + " data files.")
    elif len(arguments) < 5:
        num_plots = arguments[2]
        data_file_name = arguments[3]
        print("Using default name for json file: \"" + json_file_name + "\" and plotting " + str(num_plots) + " data files.")
    elif not os.path.exists(arguments[1]):
        print("Give a proper path name to the parent directory of the data files you want to use for plotting.")
        print("You gave \"" + str(arguments[1]) + "\", which is not a proper path name.")
        sys.exit(1)
    elif len(arguments[2]) == 0:
        print("Data file name can't be an empty string. The string you gave is \"" + arguments[2] + "\".")
        sys.exit(1)
    elif len(arguments[3]) == 0:
        print("Json file name can't be an empty string. The string you gave is \"" + arguments[3] + "\".")
        sys.exit(1)
    else:
        num_plots = arguments[2]
        data_file_name = arguments[3]
        json_file_name = arguments[4]
    
    fig = plt.figure()
    ax = fig.add_subplot(111) # nrows, ncols, index
    plot_relative_radius(ax, arguments[1], data_file_name, json_file_name, num_plots)

if __name__ == "__main__":
    main()
