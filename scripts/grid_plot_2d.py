import numpy as np
import matplotlib.pyplot as plt

def main():
    data = np.loadtxt("data/indices.dat")
    indices = np.argsort(data[:,3])

    for i in range(len(data) - 1):
        x1 = data[indices[i],0] + data[indices[i], 2] / 16.0
        y1 = data[indices[i], 1]
        x2 = data[indices[i + 1],0] + data[indices[i + 1], 2] / 16.0
        y2 = data[indices[i + 1], 1]
        plt.plot(np.array([x1, x2]), np.array([y1, y2]), 'ro-')

    plt.show()

if __name__ == "__main__":
    main()
