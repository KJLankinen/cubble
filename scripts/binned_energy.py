import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import os

def bin_snapshot(snapshot, json_file, ax):
    data = np.loadtxt(snapshot, skiprows=1, delimiter=",")

    with open(json_file, 'r') as f:
        decoded_json = json.load(f)
    phi = decoded_json["phiTarget"]

    num_bins = 10
    bin_size_x = (np.max(data[:,0]) * 1.001) / num_bins
    bin_size_y = (np.max(data[:,1]) * 1.001) / num_bins

    binned_data = np.zeros(shape=(num_bins, num_bins, 3))

    for row in data:
        i = int(row[0] / bin_size_x)
        j = int(row[1] / bin_size_y)
        binned_data[i,j,0] += row[3] * row[3] * np.pi / (bin_size_x * bin_size_y)
        binned_data[i,j,1] += row[11]
        binned_data[i,j,2] += 1

    binned_data[:,:,0] -= phi
    binned_data[:,:,1] /= binned_data[:,:,2]
    binned_data[:,:,2] /= np.max(binned_data[:,:,2])

    values, edges = np.histogram(binned_data[:,:,2].flatten(), bins='auto')
    locs = (edges[1:] - edges[0:-1:1]) / 2 + edges[0:-1:1]

    ax.plot(locs, values, 'D:', fillstyle='none', linewidth=3.0)

def main():
    loc = sys.argv[1]
    input_params = os.path.join(loc, "input_parameters.json")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(len(sys.argv) - 2):
        bin_snapshot(os.path.join(loc, f"snapshot.csv.{sys.argv[i + 2]}"), input_params, ax)

    plt.show()

if __name__ == "__main__":
    main()
