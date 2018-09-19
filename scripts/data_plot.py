import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import sys
import os
import json

##########
# Fitting the data -- Least Squares Method
##########

# Power-law fitting is best done by first converting
# to a linear equation and then fitting to a straight line.
# Note that the `logyerr` term here is ignoring a constant prefactor.
#
#  y = a * x^b
#  log(y) = log(a) + b*log(x)
#

def fit_to_data(xdata, ydata):
    logx = np.log10(xdata)
    logy = np.log10(ydata)
    yerr = np.ones(ydata.shape)
    logyerr = yerr / ydata
    
    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err
    powerlaw = lambda x, amp, index: amp * (x**index)
    
    pinit = [1.0, -1.0]
    out = optimize.leastsq(errfunc, pinit,
                           args=(logx, logy, logyerr), full_output=1)
    
    pfinal = out[0]
    covar = out[1]
    print pfinal
    print covar
    
    index = pfinal[1]
    amp = 10.0**pfinal[0]
    
    indexErr = np.sqrt( covar[1][1] )
    ampErr = np.sqrt( covar[0][0] ) * amp
    
    ##########
    # Plotting data
    ##########
    
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(xdata, powerlaw(xdata, amp, index))     # Fit
    plt.errorbar(xdata, ydata, yerr=yerr, fmt='k.')  # Data
    plt.text(5, 6.5, 'Ampli = %5.2f +/- %5.2f' % (amp, ampErr))
    plt.text(5, 5.5, 'Index = %5.2f +/- %5.2f' % (index, indexErr))
    plt.title('Best Fit Power Law')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(1, 11)

    plt.subplot(2, 1, 2)
    plt.loglog(xdata, powerlaw(xdata, amp, index))
    plt.errorbar(xdata, ydata, yerr=yerr, fmt='k.')  # Data
    plt.xlabel('X (log scale)')
    plt.ylabel('Y (log scale)')
    plt.xlim(1.0, 11)

    plt.show()

def plot_data_loglog(data_file, json_file, ax):
    data = np.loadtxt(data_file)
    x = data[:, 0]
    y = data[:, 1]
            
    with open(json_file, 'r') as f:
        decoded_json = json.load(f)
            
    phi = decoded_json["PhiTarget"]
    kappa = decoded_json["Kappa"]
    label_str = r"$\phi=$" + str(phi) + r", $\kappa=$" + str(kappa)
    
    ax.loglog(x, y, '.', linewidth=1.0, label=label_str)

def plot_relative_radius(ax, parent_dir, data_file_name, json_file_name):
    alpha1 = 0.47
    alpha2 = 0.5
    alpha3 = 0.57
    alpha4 = 0.62
    xlim = 3000
    
    ax.xaxis.label.set_fontsize(20)
    ax.yaxis.label.set_fontsize(20)
    ax.set_xlim(1, xlim)
    ax.set_ylim(0.65, 20)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\frac{R}{<R_{in}>}$")
    ax.grid(1)

    line_x1 = np.linspace(10, 600)
    #line_x2 = np.linspace(10, 600)
    #line_x3 = np.linspace(10, 600)
    line_x4 = np.linspace(10, 600)
    
    ax.loglog(line_x1, pow(line_x1 / 11.5, alpha1), 'k--', linewidth=2.0, label=r"$a\tau^b, b=$" + str(alpha1))
    #ax.loglog(line_x2, pow(line_x2 / 12.5, alpha2), 'r--', linewidth=2.0, label=r"$a\tau^b, b=$" + str(alpha2))
    #ax.loglog(line_x3, pow(line_x3 / 16., alpha3), 'g--', linewidth=2.0, label=r"$a\tau^b, b=$" + str(alpha3))
    ax.loglog(line_x4, pow(line_x4 / 19.0, alpha4), 'r--', linewidth=2.0, label=r"$a\tau^b, b=$" + str(alpha4))

    child_dirs = [name for name in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, name))]
    json_files = []
    data_files = []
    phis = []

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
                json_files.append(json_file)
                data_files.append(data_file)

    indices = list(reversed(np.argsort(np.array(phis))))
    json_files = [json_files[idx] for idx in indices]
    data_files = [data_files[idx] for idx in indices]

    # Plot
    for i in range(len(data_files)):
        plot_data_loglog(data_files[i], json_files[i], ax)

    ax.legend(loc=2, prop={'size' : 20})

    plt.show()
    
def main():
    arguments = sys.argv
    data_file_name = ""
    json_file_name = ""
    
    if len(arguments) < 2:
        print("Too few arguments given. Give:\n\tthe parent folder of data files\n\tname of data file\n\tname of saved json file\nas arguments.")
        sys.exit(1)
    elif len(arguments) < 3:
        data_file_name = "collected_data.dat"
        json_file_name = "save.json"
        print("Using default names for data and json files, \"" + data_file_name + "\" and \"" + json_file_name + "\" respectively.")
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
        data_file_name = arguments[2]
        json_file_name = arguments[3]
    
    fig = plt.figure()
    ax = fig.add_subplot(111) # nrows, ncols, index
    plot_relative_radius(ax, arguments[1], data_file_name, json_file_name)

if __name__ == "__main__":
    main()
