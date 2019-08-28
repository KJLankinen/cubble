import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import json

def plot_averages(directory, filename_stem, index):
    averages = []
    std = []
    var = []
    t = []

    for root, dirs, files in os.walk(directory):
        for name in files:
            if filename_stem in name:
                data = np.loadtxt(os.path.join(root, name), delimiter=',', skiprows=1)
                n = len(data[:,index])
                avg = np.sum(data[:,index]) / n
                averages.append(avg)
                diff = data[:,index] - avg
                variance = np.sum(diff**2) / n
                var.append(variance)
                std.append(np.sqrt(variance))
                snapshot_num = float(name[name.rfind('.') + 1:])
                t.append(snapshot_num)

    averages = np.array(averages)
    std = np.array(std)
    var = np.array(var)
    t = np.array(t)

    with open(os.path.join(directory, "input_parameters.json"), 'r') as f:
        decoded_json = json.load(f)
            
    freq = decoded_json["snapshotFrequency"]
    t = t / freq

    inds = t.argsort()
    t = t[inds]
    averages = averages[inds]
    std = std[inds]
    var = var[inds]
    
    data = np.loadtxt(os.path.join(directory, "results.dat"))
    
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #fig, ax = plt.subplots(2, 1, sharex=True)
    fig = plt.figure()
    ax = fig.add_subplot()

    start_index = 14000

    lns1 = ax.plot(data[start_index:,0], data[start_index:,5], linewidth=5.0, label="E")
    #ax.set_xlim(, 1000)
    
    ax.xaxis.label.set_fontsize(50)
    ax.xaxis.set_label_coords(0.035, 0.055)
    ax.yaxis.label.set_fontsize(50)
    ax.yaxis.set_label_coords(0.035, 0.87)

    ax.tick_params(axis='x', which='both', labelsize=40, direction='in', pad=-50)
    ax.tick_params(axis='y', which='both', labelsize=38)
    ax.set_xlabel(r"$\tau$")

    ax_0 = ax.twinx()
    ax_0.set_ylim(-1.5, 1.5)
    ax_0.tick_params(axis='y', which='both', labelsize=40)
    snapshot_start_index = np.argmax(t > start_index)
    lns2 = ax_0.plot(t[snapshot_start_index:], var[snapshot_start_index:] / np.max(var[snapshot_start_index:]),
            color=color_cycle[1],
            linewidth=2.5,
            label=r"$\sigma^2_{vel}$",
            alpha=0.75)
    lns3 = ax_0.plot(data[start_index:,0],
            np.clip(data[start_index:,6], np.min(data[start_index:,6]), 0) / -np.min(data[start_index:]),
            color=color_cycle[2],
            linewidth=2.5,
            label=r"$\Delta E$",
            alpha=0.75)

    #peaks = 0.2 * ((var - np.min(var)) / (np.max(var[100:]) - np.min(var)))[0:-2:1] +\
    #        0.8 * ((data[:,6] - np.max(data[:,6])) / (np.min(data[100:,6]) - np.max(data[:,6])))
    #peaks = (peaks - np.min(peaks[100:])) / (np.max(peaks[100:]) - np.min(peaks[100:]))

    #inds = np.greater(peaks, 0.25)
    #for xc in data[100:,0][inds[100:]]:
    #    ax[1].axvline(x=xc)
    #ax[1].grid(1)
    #ax[1].twinx().plot(data[:,0], peaks, color=color_cycle[1])
    
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper right', fontsize=30)
    plt.subplots_adjust(left=0.07, right=0.93, top=0.99, bottom=0.01)
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Provide the directory containing the snapshots.")

    filename_stem = "snapshot.csv."
    plot_averages(sys.argv[1], filename_stem, 7)

if __name__ == "__main__":
    main()
