import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import os

def bin_snapshot(snapshot, json_file):
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
    plt.hist(binned_data[:,:,0].flatten())
    plt.show()

def main():
    loc = sys.argv[1]
    snapshot = os.path.join(loc, f"snapshot.csv.{sys.argv[2]}")
    input_params = os.path.join(loc, "input_parameters.json")
    bin_snapshot(snapshot, input_params)

if __name__ == "__main__":
    main()
