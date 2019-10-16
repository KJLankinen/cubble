import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def main():
    root_dir = sys.argv[1]
    f = plt.figure()
    ax = plt.subplot(111)

    for i in ["phi_876", "phi_865", "phi_852", "phi_851"]:
        data = np.loadtxt(os.path.join(root_dir, i, "snapshot.csv.0"), delimiter=',', skiprows=1)
        #values, edges = np.histogram(data[:,11] / np.max(data[:,11]), bins='auto')
        #locs = (edges[1:] - edges[0:-1:1]) / 2 + edges[0:-1:1]
        #ax.semilogy(locs, values, ':', fillstyle='none', linewidth=1.0, label=i)

        ax.hist(data[:,11] / np.max(data[:,11]), bins='auto', alpha=0.33)

    plt.show()

if __name__ == "__main__":
    main()
