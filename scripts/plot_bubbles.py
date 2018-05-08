import numpy as np
import matplotlib.pyplot as plt

def main():
    data = np.loadtxt("./data/bubbles.dat", delimiter=',')
    x = data[:,0]
    y = data[:,1]
    r = data[:,3]

    plt.scatter(x, y, s=r*100)
    plt.show()

if __name__ == "__main__":
    main()
