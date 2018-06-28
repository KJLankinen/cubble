'''
This script divides N values to pairs in four domains s.t. the domain sizes are as equal to
each other as possible and, more importantly, each domain accesses only as few different
indices as possible.

The motivation behind this is the very limited amount of shared memory cuda has available
for each thread block. If all bubble-pairs can be computed in such a way that each
thread block is able to load all of its bubbles to shared memory, and the number of pairs
is divided evenly among the thread blocks, then the work to be done is shared optimally
and the memory used is the fastest available.

Note that this script doesn't have anything to do with the cuda or c++ implementation,
this is only a visual aid for the pair-wise domain division.
'''

import numpy as np
import matplotlib.pyplot as plt

def main():
    n = 96
    values = np.zeros((n, n))
    
    # First domain
    s1 = set()
    s = 2 * int(n / 4) + 1
    i = s
    d1 = 0
    while (i < n):
        s1.add(i)
        j = s - 1
        while (j < i):
            s1.add(j)
            values[i, j] = 1
            j += 1
            d1 += 1
        i += 1

    # Second domain
    s2 = set()
    i = 5 * n / 8
    d2 = 0
    while (i < n):
        s2.add(i)
        j = 0
        while (j < n/4):
            s2.add(j)
            values[i, j] = 2
            j += 1
            d2 += 1
        i += 1

    i = 1
    while (i < n/4):
        s2.add(i)
        j = 0
        while (j < i):
            s2.add(j)
            values[i, j] = 2
            j += 1
            d2 += 1
        i += 1

    # Third domain
    s3 = set()
    i = 5 * n / 8
    d3 = 0
    while (i < n):
        s3.add(i)
        j = n / 4
        while (j < 2 * n/4):
            s3.add(j)
            values[i, j] = 3
            j += 1
            d3 += 1
        i += 1

    i = n / 4
    while (i < 2 * n / 4):
        s3.add(i)
        j = n / 4
        while (j < i):
            s3.add(j)
            values[i, j] = 3
            j += 1
            d3 += 1
        i += 1

    # Fourth domain
    s4 = set()
    i = n / 4
    d4 = 0
    while (i < 5 * n / 8):
        s4.add(i)
        j = 0
        while (j < n / 4):
            s4.add(j)
            values[i, j] = 4
            j += 1
            d4 += 1
        i += 1

    i = 2 * n / 4
    while (i < 5 * n / 8):
        s4.add(i)
        j = n / 4
        while (j < 2 * n / 4):
            s4.add(j)
            values[i, j] = 4
            j += 1
            d4 += 1
        i += 1

    print("Number of elements in domain 1: " + str(d1) + ", number of unique indices: " + str(len(s1)))
    print("Number of elements in domain 2: " + str(d2) + ", number of unique indices: " + str(len(s2)))
    print("Number of elements in domain 3: " + str(d3) + ", number of unique indices: " + str(len(s3)))
    print("Number of elements in domain 4: " + str(d4) + ", number of unique indices: " + str(len(s4)))
    print("Total: " + str(d1 + d2 + d3 + d4))
    print("Total num: " + str(n * (n - 1) / 2))
    
    plt.imshow(values, interpolation="none", extent=(0, n, n, 0))
    plt.grid(1)
    plt.show()

if __name__ == "__main__":
    main()
