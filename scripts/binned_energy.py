import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import os

def bin_snapshot(snapshot, json_file, ax, label):
    data = np.loadtxt(snapshot, skiprows=1, delimiter=",")

    with open(json_file, 'r') as f:
        decoded_json = json.load(f)
    phi = decoded_json["phiTarget"]

    num_bins = 10
    bin_size_x = (np.max(data[:,0]) * 1.001) / num_bins
    bin_size_y = (np.max(data[:,1]) * 1.001) / num_bins

    binned_data = np.zeros(shape=(num_bins, num_bins, 4))

    for row in data:
        i = int(row[0] / bin_size_x)
        j = int(row[1] / bin_size_y)
        binned_data[i,j,0] += row[3] * row[3] * np.pi / (bin_size_x * bin_size_y)
        binned_data[i,j,1] += row[11]
        binned_data[i,j,2] += 1
        binned_data[i,j,3] += row[3]

    phi_dev = binned_data[:,:,0] / np.sum(binned_data[:,:,0]) * num_bins * num_bins
    energy = binned_data[:,:,1] / np.sum(binned_data[:,:,1]) * num_bins * num_bins
    num_bubbles = binned_data[:,:,2] / np.sum(binned_data[:,:,2]) * num_bins * num_bins
    avg_rad = binned_data[:,:,3] / np.sum(binned_data[:,:,3]) * num_bins * num_bins

    values, edges = np.histogram(energy.flatten() - 1.0, bins='auto')
    locs = (edges[1:] - edges[0:-1:1]) / 2 + edges[0:-1:1]

    ax.plot(locs, values, 'D:', fillstyle='none', linewidth=3.0, label=label)

def bin_evolution(from_t, to_t, directory, input_params, num_bins, axes1, axes2):
    with open(input_params, 'r') as f: decoded_json = json.load(f)
    phi = decoded_json["phiTarget"]
    freq = decoded_json["snapshotFrequency"]

    num_values = 4
    binned_data = np.zeros(shape=(num_bins, num_bins, num_values, to_t - from_t))
    time = []
    avgs = np.zeros(shape=(num_values, to_t - from_t))
    stds = np.zeros(shape=(num_values, to_t - from_t))
    for t in range(to_t - from_t):
        snapshot = os.path.join(directory, f"snapshot.csv.{from_t + t}")
        data = np.loadtxt(snapshot, skiprows=1, delimiter=",")
        time.append((from_t + t) / freq)
        bin_size_x = (np.max(data[:,0]) * 1.001) / num_bins
        bin_size_y = (np.max(data[:,1]) * 1.001) / num_bins
        for row in data:
            i = int(row[0] / bin_size_x)
            j = int(row[1] / bin_size_y)
            binned_data[i,j,0,t] += row[3] * row[3] * np.pi / (bin_size_x * bin_size_y)
            binned_data[i,j,1,t] += row[11]
            binned_data[i,j,2,t] += 1
            binned_data[i,j,3,t] += row[3]

        for i in range(num_values):
            binned_data[:,:,i,t] = binned_data[:,:,i,t] / np.sum(binned_data[:,:,i,t]) * num_bins * num_bins
            avgs[i,t] = np.sum(binned_data[:,:,i,t]) / float(len(binned_data[:,:,i,t].flatten()))
            stds[i,t] = np.sum((avgs[i,t] - binned_data[:,:,i,t])**2) / len(binned_data[:,:,i,t].flatten())

    idx = 0
    time = np.array(time)
    for i in range(num_bins):
        for j in range(num_bins):
            axes1[num_bins - 1 - j, i].plot(time, binned_data[i,j,idx,:] - 1.0)

    axes2[0].plot(time, avgs[idx])
    axes2[1].plot(time, stds[idx])

def main():
    loc = sys.argv[1]
    fig = plt.figure()
    #dirs = ["phi_840", "phi_850", "phi_860","phi_876"]
    dirs = ["phi_865","phi_866"]
    #dirs = ["phi_864"]

    if sys.argv[2] == "self" or sys.argv[2] == "peer":
        snaps = sys.argv[3:]
        ncols = 2
        if sys.argv[2] == "self":
            nrows = int(np.ceil(len(dirs) / ncols)) 
            axes = fig.subplots(nrows, ncols, squeeze=False)

            for i, directory in enumerate(dirs):
                ax = axes[int(i / ncols), int(i % ncols)]
                title = fr"$\phi$ 0.{directory[-3:]}"
                ax.set_title(title)

                for snap in snaps:
                    input_params = os.path.join(loc, directory, "input_parameters.json")

                    with open(input_params, 'r') as f:
                        decoded_json = json.load(f)
                    freq = decoded_json["snapshotFrequency"]

                    snapshot = os.path.join(loc, directory, f"snapshot.csv.{snap}")
                    label = fr"$\tau$ {int(snap) / freq}"

                    bin_snapshot(snapshot, input_params, ax, label) 

                ax.legend()
        elif sys.argv[2] == "peer":
            nrows = int(np.ceil(len(snaps) / ncols)) 
            axes = fig.subplots(nrows, ncols, squeeze=False)

            for i, snap in enumerate(snaps):
                ax = axes[int(i / ncols), int(i % ncols)]
                
                for directory in dirs:
                    input_params = os.path.join(loc, directory, "input_parameters.json")

                    with open(input_params, 'r') as f:
                        decoded_json = json.load(f)
                    freq = decoded_json["snapshotFrequency"]

                    title = fr"$\tau$ {int(snap) / freq}"
                    snapshot = os.path.join(loc, directory, f"snapshot.csv.{snap}")
                    label = fr"$\phi$ 0.{directory[-3:]}"

                    bin_snapshot(snapshot, input_params, ax, label)

                ax.set_title(title)
                ax.legend()
    elif sys.argv[2] == "time":
        from_t = int(sys.argv[3])
        to_t = int(sys.argv[4])
        num_bins = 4
        axes1 = fig.subplots(num_bins, num_bins, sharex=True, sharey=True)
        fig2 = plt.figure()
        axes2 = fig2.subplots(2, 1) 
        for phi in dirs:
            directory = os.path.join(loc, phi)
            input_params = os.path.join(directory, "input_parameters.json")
            bin_evolution(from_t, to_t, directory, input_params, num_bins, axes1, axes2)
    else:
        print(f"Bad input arguments: {sys.argv}")
        return 1

    plt.show()

if __name__ == "__main__":
    main()
