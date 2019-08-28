import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_de(file_numbers, dirname, filename):
    fig, ax = plt.subplots(len(file_numbers), 1, sharex=True)

    for i, n in enumerate(file_numbers):
        data = np.loadtxt(f"{dirname}{n}/{filename}")
        ax[i].set_xlim(10, 900)
        ax[i].set_ylim(-0.02, 0.02)
        ax[i].set_xlabel(r"$\tau$")
        ax[i].set_ylabel(r"$\frac{\langle R \rangle}{\langle R_{in} \rangle}$", rotation=0)
        ax[i].plot(data[5:,0], data[5:,6])
        ax[i].grid(1)

    plt.show()

def plot_e(file_numbers, dirname, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\frac{\langle R \rangle}{\langle R_{in} \rangle}$", rotation=0)

    for i, n in enumerate(file_numbers):
        data = np.loadtxt(f"{dirname}{n}/{filename}")
        #data = data[np.less(data[:,2], data[0,2] * 0.01)]
        #ax.semilogx(data[5:,0], data[5:,5] / np.max(data[5:,5]))
        ax.plot(data[5:,0], data[5:,5], linewidth=5.0)
        ax.axvline(x=30, linewidth=3.0)
        ax.axvline(x=160, linewidth=3.0)
        ax.axvline(x=400, linewidth=3.0)

    ax.xaxis.label.set_fontsize(50)
    ax.xaxis.set_label_coords(0.035, 0.055)
    ax.yaxis.label.set_fontsize(50)
    ax.yaxis.set_label_coords(0.035, 0.87)

    ax.tick_params(axis='x', which='both', labelsize=40, direction='in', pad=-50)
    ax.tick_params(axis='y', which='both', labelsize=38, direction='in', pad=-50)
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("E")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    plt.show()

def main():
    file_numbers = []
    for i in range(len(sys.argv) - 1):
        file_numbers.append(sys.argv[i + 1])

    dirname = "run_"
    filename = "results.dat"
    plot_e(file_numbers, dirname, filename)

if __name__ == "__main__":
    main()
