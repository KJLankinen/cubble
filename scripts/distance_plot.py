import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def plot_snapshots(data_dir, first_snapshot, last_snapshot, ax):
    for i in range(first_snapshot, last_snapshot + 1):
        filename = data_dir + "/snapshot.csv." + str(i)
        if not os.path.isfile(filename):
            print("\"" + filename + "\" is not a valid filename.")
            return 1
        data = np.loadtxt(filename, delimiter=",")
        ax.loglog(data[:, -2], data[:, -1], '.')

def plot_avg(data_dir, ax):
    filename = data_dir + "/output.dat"
    if not os.path.isfile(filename):
        print("Invalid filename for data file containing the averages of the run.")
        print("\"" + filename + "\" is not a valid filename.")
        return 1

    data = np.loadtxt(filename)

    ax.loglog(data[:, 0], data[:, -1])

def main():
    if (len(sys.argv) < 2):
        print("Give arguments:")
        print("1: Data directory.")
        #print("2: First snapshot (a number).")
        #print("3: Last snapshot (a number).")
        return 1

    data_dir = sys.argv[1]
    #first_snapshot = int(sys.argv[2])
    #last_snapshot = int(sys.argv[3])
    print("Using " + data_dir + " as data directory.")
    #print("Using snapshots in range " + str(first_snapshot) + "-" + str(last_snapshot))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.set_xlabel("path length s")
    #ax.set_ylabel("distance r²")
    ax.grid(1)
    
    #plot_snapshots(data_dir, first_snapshot, last_snapshot, ax)
    plot_avg(data_dir, ax)

    x = np.linspace(1.5, 1000, 1000)
    y = pow(x, 1.2) * 0.25
    ax.loglog(x, y)

    plt.show()

    
if __name__ == "__main__":
    main()