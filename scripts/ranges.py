import numpy as np
import matplotlib.pyplot as plt

def main():
    data = np.loadtxt("ranges.csv", delimiter=',')
    starts = data[:,0]
    widths = data[:,1]
    colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, 6))

    f = plt.figure()
    ax = f.add_subplot()
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    for i in range(len(widths)):
        col = colors[int(i / 8)]
        col[3] = 0.8
        ax.barh(0, widths[i], left=starts[i], color=col)
    plt.show()

if __name__ == "__main__":
    main()
