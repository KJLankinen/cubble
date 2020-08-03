#include "Kernels.cuh"

namespace cubble {
__device__ double dTotalArea;
__device__ double dTotalOverlapArea;
__device__ double dTotalOverlapAreaPerRadius;
__device__ double dTotalAreaPerRadius;
__constant__ __device__ double dTotalVolume;
__device__ bool dErrorEncountered;
__device__ int dNumPairs;
__device__ int dNumBubblesAboveMinRad;
__device__ double dVolumeMultiplier;

__device__ void logError(bool condition, const char *statement,
                         const char *errMsg) {
    if (condition == false) {
        printf("----------------------------------------------------"
               "\nError encountered"
               "\n(%s) -> %s"
               "\n@thread[%d, %d, %d], @block[%d, %d, %d]"
               "\n----------------------------------------------------\n",
               statement, errMsg, threadIdx.x, threadIdx.y, threadIdx.z,
               blockIdx.x, blockIdx.y, blockIdx.z);

        dErrorEncountered = true;
    }
}

__device__ int getGlobalTid() {
    // Simple helper function for calculating a 1D coordinate
    // from 1, 2 or 3 dimensional coordinates.
    int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
    int blocksBefore = blockIdx.z * (gridDim.y * gridDim.x) +
                       blockIdx.y * gridDim.x + blockIdx.x;
    int threadsBefore =
        blockDim.y * blockDim.x * threadIdx.z + blockDim.x * threadIdx.y;
    int tid = blocksBefore * threadsPerBlock + threadsBefore + threadIdx.x;

    return tid;
}

__device__ void resetDoubleArrayToValue(double value, int idx, double *array) {
    array[idx] = value;
}

__device__ void setFlagIfLessThanConstant(int idx, int *flags, double *values,
                                          double constant) {
    flags[idx] = values[idx] < constant ? 1 : 0;
}

__device__ void setFlagIfGreaterThanConstant(int idx, int *flags,
                                             double *values, double constant) {
    flags[idx] = values[idx] > constant ? 1 : 0;
}

__device__ dvec wrappedDifference(dvec p1, dvec p2, dvec interval) {
    const dvec d1 = p1 - p2;
    dvec d2 = d1;
    dvec temp = interval - d1.getAbsolute();
#if (PBC_X == 1)
    if (temp.x * temp.x < d1.x * d1.x) {
        d2.x = temp.x * (d1.x < 0 ? 1.0 : -1.0);
    }
#endif
#if (PBC_Y == 1)
    if (temp.y * temp.y < d1.y * d1.y) {
        d2.y = temp.y * (d1.y < 0 ? 1.0 : -1.0);
    }
#endif
#if (PBC_Z == 1)
    if (temp.z * temp.z < d1.z * d1.z) {
        d2.z = temp.z * (d1.z < 0 ? 1.0 : -1.0);
    }
#endif

    return d2;
}

__global__ void transformPositionsKernel(bool normalize, int numValues,
                                         dvec lbb, dvec tfr, double *x,
                                         double *y, double *z) {
    const dvec interval = tfr - lbb;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues;
         i += gridDim.x * blockDim.x) {
        if (normalize) {
            x[i] = (x[i] - lbb.x) / interval.x;
            y[i] = (y[i] - lbb.y) / interval.y;
#if (NUM_DIM == 3)
            z[i] = (z[i] - lbb.z) / interval.z;
#endif
        } else {
            x[i] = interval.x * x[i] + lbb.x;
            y[i] = interval.y * y[i] + lbb.y;
#if (NUM_DIM == 3)
            z[i] = interval.z * z[i] + lbb.z;
#endif
        }
    }
}

__device__ int getNeighborCellIndex(ivec cellIdx, ivec dim, int neighborNum) {
    ivec idxVec = cellIdx;
    switch (neighborNum) {
    case 0:
        // self
        break;
    case 1:
        idxVec += ivec(-1, 1, 0);
        break;
    case 2:
        idxVec += ivec(-1, 0, 0);
        break;
    case 3:
        idxVec += ivec(-1, -1, 0);
        break;
    case 4:
        idxVec += ivec(0, -1, 0);
        break;
    case 5:
        idxVec += ivec(-1, 1, -1);
        break;
    case 6:
        idxVec += ivec(-1, 0, -1);
        break;
    case 7:
        idxVec += ivec(-1, -1, -1);
        break;
    case 8:
        idxVec += ivec(0, 1, -1);
        break;
    case 9:
        idxVec += ivec(0, 0, -1);
        break;
    case 10:
        idxVec += ivec(0, -1, -1);
        break;
    case 11:
        idxVec += ivec(1, 1, -1);
        break;
    case 12:
        idxVec += ivec(1, 0, -1);
        break;
    case 13:
        idxVec += ivec(1, -1, -1);
        break;
    default:
        printf("Should never end up here!\n");
        break;
    }

#if (PBC_X == 1)
    idxVec.x += dim.x;
    idxVec.x %= dim.x;
#else
    if (idxVec.x < 0 || idxVec.x >= dim.x)
        return -1;
#endif

#if (PBC_Y == 1)
    idxVec.y += dim.y;
    idxVec.y %= dim.y;
#else
    if (idxVec.y < 0 || idxVec.y >= dim.y)
        return -1;
#endif

#if (PBC_Z == 1)
    idxVec.z += dim.z;
    idxVec.z %= dim.z;
#else
    if (idxVec.z < 0 || idxVec.z >= dim.z)
        return -1;
#endif

    return get1DIdxFrom3DIdx(idxVec, dim);
}

__device__ double getWrappedCoordinate(double val1, double val2,
                                       double multiplier) {
    double difference = val1 - val2;
    val2 = difference < -0.5 * multiplier
               ? val2 - multiplier
               : (difference > 0.5 * multiplier ? val2 + multiplier : val2);
    val2 = val1 - val2;

    return val2;
}

__device__ int getCellIdxFromPos(double x, double y, double z, dvec lbb,
                                 dvec tfr, ivec cellDim) {
    const dvec interval = tfr - lbb;
    const int xid = floor(cellDim.x * (x - lbb.x) / interval.x);
    const int yid = floor(cellDim.y * (y - lbb.y) / interval.y);
#if (NUM_DIM == 3)
    const int zid = floor(cellDim.z * (z - lbb.z) / interval.z);
#else
    const int zid = 0;
#endif

    return get1DIdxFrom3DIdx(ivec(xid, yid, zid), cellDim);
}

__device__ __host__ int get1DIdxFrom3DIdx(ivec idxVec, ivec cellDim) {
// Linear encoding
// return idxVec.z * cellDim.x * cellDim.y + idxVec.y * cellDim.x + idxVec.x;

// Morton encoding
#if (NUM_DIM == 3)
    return encodeMorton3((unsigned int)idxVec.x, (unsigned int)idxVec.y,
                         (unsigned int)idxVec.z);
#else
    return encodeMorton2((unsigned int)idxVec.x, (unsigned int)idxVec.y);
#endif
}

__device__ __host__ ivec get3DIdxFrom1DIdx(int idx, ivec cellDim) {
    ivec idxVec(0, 0, 0);
// Linear decoding
/*
idxVec.x = idx % cellDim.x;
idxVec.y = (idx / cellDim.x) % cellDim.y;
#if (NUM_DIM == 3)
idxVec.z = idx / (cellDim.x * cellDim.y);
#endif
*/
#if (NUM_DIM == 3)
    idxVec.x = decodeMorton3x((unsigned int)idx);
    idxVec.y = decodeMorton3y((unsigned int)idx);
    idxVec.z = decodeMorton3z((unsigned int)idx);
#else
    idxVec.x = decodeMorton2x((unsigned int)idx);
    idxVec.y = decodeMorton2y((unsigned int)idx);
#endif

    return idxVec;
}

__device__ __host__ unsigned int encodeMorton2(unsigned int x, unsigned int y) {
    return (part1By1(y) << 1) + part1By1(x);
}

__device__ __host__ unsigned int encodeMorton3(unsigned int x, unsigned int y,
                                               unsigned int z) {
    return (part1By2(z) << 2) + (part1By2(y) << 1) + part1By2(x);
}

__device__ __host__ unsigned int decodeMorton2x(unsigned int code) {
    return compact1By1(code >> 0);
}

__device__ __host__ unsigned int decodeMorton2y(unsigned int code) {
    return compact1By1(code >> 1);
}

__device__ __host__ unsigned int decodeMorton3x(unsigned int code) {
    return compact1By2(code >> 0);
}

__device__ __host__ unsigned int decodeMorton3y(unsigned int code) {
    return compact1By2(code >> 1);
}

__device__ __host__ unsigned int decodeMorton3z(unsigned int code) {
    return compact1By2(code >> 2);
}

__device__ __host__ unsigned int part1By1(unsigned int x) {
    // Mask the lowest 16 bits
    x &= 0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x ^ (x << 8)) &
        0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x << 4)) &
        0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x << 2)) &
        0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x << 1)) &
        0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0

    return x;
}

__device__ __host__ unsigned int part1By2(unsigned int x) {
    // Mask lowest 10 bits
    x &= 0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210
    x = (x ^ (x << 16)) &
        0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x << 8)) &
        0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x << 4)) &
        0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x << 2)) &
        0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0

    return x;
}

__device__ __host__ unsigned int compact1By1(unsigned int x) {
    x &= 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    x = (x ^ (x >> 1)) &
        0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x >> 2)) &
        0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x >> 4)) &
        0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x >> 8)) &
        0x0000ffff; // x = ---- ---- ---- ---- fedc ba98 7654 3210
    return x;
}

__device__ __host__ unsigned int compact1By2(unsigned int x) {
    x &= 0x09249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
    x = (x ^ (x >> 2)) &
        0x030c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x ^ (x >> 4)) &
        0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x ^ (x >> 8)) &
        0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x ^ (x >> 16)) &
        0x000003ff; // x = ---- ---- ---- ---- ---- --98 7654 3210

    return x;
}

__device__ void comparePair(int idx1, int idx2, double *r, int *first,
                            int *second, dvec interval, double *x, double *y,
                            double *z) {
    const double r1 = r[idx1];
    const double r2 = r[idx2];
    double maxDistance = r1 > r2 ? r1 : r2;
    maxDistance *= 2.5;
    maxDistance += (r1 < r2) ? r1 : r2;
    dvec p1 = dvec(x[idx1], y[idx1], 0.0);
    dvec p2 = dvec(x[idx2], y[idx2], 0.0);
#if (NUM_DIM == 3)
    p1.z = z[idx1];
    p2.z = z[idx2];
#endif
    if (wrappedDifference(p1, p2, interval).getSquaredLength() <
        maxDistance * maxDistance) {
        // Set the smaller idx to idx1 and larger to idx2
        int id = idx1 > idx2 ? idx1 : idx2;
        idx1 = idx1 < idx2 ? idx1 : idx2;
        idx2 = id;

        id = atomicAdd(&dNumPairs, 1);
        first[id] = idx1;
        second[id] = idx2;
    }
}

__global__ void wrapKernel(int numValues, dvec lbb, dvec tfr, double *x,
                           double *y, double *z, int *mx, int *my, int *mz,
                           int *mpx, int *mpy, int *mpz) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues;
         i += gridDim.x * blockDim.x) {
        double *v;
        int *m;
        int *mp;
        double value;
        double low;
        double high;
        int mult;
#if (PBC_X == 1)
        v = x;
        m = mx;
        mp = mpx;
        value = v[i];
        low = lbb.x;
        high = tfr.x;
        mult = value < low ? 1 : (value > high ? -1 : 0);

        v[i] = value + (high - low) * (double)mult;
        m[i] = mp[i] - mult;
#endif
#if (PBC_Y == 1)
        v = y;
        m = my;
        mp = mpy;
        value = v[i];
        low = lbb.y;
        high = tfr.y;
        mult = value < low ? 1 : (value > high ? -1 : 0);

        v[i] = value + (high - low) * (double)mult;
        m[i] = mp[i] - mult;
#endif
#if (PBC_Z == 1)
        v = z;
        m = mz;
        mp = mpz;
        value = v[i];
        low = lbb.z;
        high = tfr.z;
        mult = value < low ? 1 : (value > high ? -1 : 0);

        v[i] = value + (high - low) * (double)mult;
        m[i] = mp[i] - mult;
#endif
    }
}

__global__ void calculateVolumes(double *r, double *volumes, int numValues) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues;
         i += gridDim.x * blockDim.x) {
        double radius = r[i];
        double volume = radius * radius * CUBBLE_PI;
#if (NUM_DIM == 3)
        volume *= radius * 1.33333333333333333333333333;
#endif

        volumes[i] = volume;
    }
}

__global__ void assignDataToBubbles(double *x, double *y, double *z,
                                    double *xPrd, double *yPrd, double *zPrd,
                                    double *r, double *w, int *aboveMinRadFlags,
                                    int *indices, ivec bubblesPerDim, dvec tfr,
                                    dvec lbb, double avgRad, double minRad,
                                    int numValues) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues;
         i += gridDim.x * blockDim.x) {
        indices[i] = i;
        dvec pos(0, 0, 0);
        pos.x = (i % bubblesPerDim.x) / (double)bubblesPerDim.x;
        pos.y =
            ((i / bubblesPerDim.x) % bubblesPerDim.y) / (double)bubblesPerDim.y;

        dvec randomOffset(x[i], y[i], 0);
#if (NUM_DIM == 3)
        randomOffset.z = z[i];
        pos.z =
            (i / (bubblesPerDim.x * bubblesPerDim.y)) / (double)bubblesPerDim.z;
#endif
        dvec interval = tfr - lbb;
        pos *= interval;
        randomOffset = dvec::normalize(randomOffset) * avgRad * w[i];
        pos += randomOffset;

        r[i] = r[i] > 0 ? r[i] : -r[i];

        x[i] = pos.x > lbb.x ? (pos.x < tfr.x ? pos.x : pos.x - interval.x)
                             : pos.x + interval.x;
        y[i] = pos.y > lbb.y ? (pos.y < tfr.y ? pos.y : pos.y - interval.y)
                             : pos.y + interval.y;
        z[i] = pos.z > lbb.z ? (pos.z < tfr.z ? pos.z : pos.z - interval.z)
                             : pos.z + interval.z;

        xPrd[i] = pos.x;
        yPrd[i] = pos.y;
        zPrd[i] = pos.z;

        w[i] = 2.0 * CUBBLE_PI * r[i] / numValues;

#if (NUM_DIM == 3)
        w[i] *= 2.0 * r[i];
#endif

        setFlagIfGreaterThanConstant(i, aboveMinRadFlags, r, minRad);
    }
}

__global__ void assignBubblesToCells(double *x, double *y, double *z,
                                     int *cellIndices, int *bubbleIndices,
                                     dvec lbb, dvec tfr, ivec cellDim,
                                     int numValues) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues;
         i += gridDim.x * blockDim.x) {
        cellIndices[i] = getCellIdxFromPos(x[i], y[i], z[i], lbb, tfr, cellDim);
        bubbleIndices[i] = i;
    }
}

__global__ void neighborSearch(int neighborCellNumber, int numValues,
                               int numCells, int numMaxPairs, int *offsets,
                               int *sizes, int *first, int *second, double *r,
                               dvec interval, double *x, double *y, double *z) {
    const ivec idxVec(blockIdx.x, blockIdx.y, blockIdx.z);
    const ivec dimVec(gridDim.x, gridDim.y, gridDim.z);
    const int cellIdx2 =
        getNeighborCellIndex(idxVec, dimVec, neighborCellNumber);

    if (cellIdx2 >= 0) {
        const int cellIdx1 = get1DIdxFrom3DIdx(idxVec, dimVec);
        DEVICE_ASSERT(cellIdx1 < numCells, "Invalid cell index!");
        DEVICE_ASSERT(cellIdx2 < numCells, "Invalid cell index!");

        if (sizes[cellIdx1] == 0 || sizes[cellIdx2] == 0)
            return;

        // Self comparison only loops the upper triangle of values (n * (n - 1))
        // / 2 comparisons instead of n^2.
        if (cellIdx1 == cellIdx2) {
            const int size = sizes[cellIdx1];
            const int offset = offsets[cellIdx1];
            for (int k = threadIdx.x; k < (size * (size - 1)) / 2;
                 k += blockDim.x) {
                int idx1 =
                    size - 2 -
                    (int)floor(
                        sqrt(-8.0 * k + 4 * size * (size - 1) - 7) * 0.5 - 0.5);
                const int idx2 = offset + k + idx1 + 1 - size * (size - 1) / 2 +
                                 (size - idx1) * ((size - idx1) - 1) / 2;
                idx1 += offset;

                DEVICE_ASSERT(idx1 < numValues, "Invalid bubble index!");
                DEVICE_ASSERT(idx2 < numValues, "Invalid bubble index!");
                DEVICE_ASSERT(idx1 != idx2, "Invalid bubble index!");

                comparePair(idx1, idx2, r, first, second, interval, x, y, z);
                DEVICE_ASSERT(numMaxPairs > dNumPairs,
                              "Too many neighbor indices!");
            }
        } else // Compare all values of one cell to all values of other cell,
               // resulting in n1 * n2
               // comparisons.
        {
            const int size1 = sizes[cellIdx1];
            const int size2 = sizes[cellIdx2];
            const int offset1 = offsets[cellIdx1];
            const int offset2 = offsets[cellIdx2];
            for (int k = threadIdx.x; k < size1 * size2; k += blockDim.x) {
                const int idx1 = offset1 + k / size2;
                const int idx2 = offset2 + k % size2;

                DEVICE_ASSERT(idx1 < numValues, "Invalid bubble index!");
                DEVICE_ASSERT(idx2 < numValues, "Invalid bubble index!");
                DEVICE_ASSERT(idx1 != idx2, "Invalid bubble index!");

                comparePair(idx1, idx2, r, first, second, interval, x, y, z);
                DEVICE_ASSERT(numMaxPairs > dNumPairs,
                              "Too many neighbor indices!");
            }
        }
    }
}

__global__ void velocityPairKernel(double fZeroPerMuZero, int *pair1,
                                   int *pair2, double *r, dvec interval,
                                   double *x, double *y, double *z, double *vx,
                                   double *vy, double *vz) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs;
         i += gridDim.x * blockDim.x) {
        int idx1 = pair1[i];
        int idx2 = pair2[i];
        double radii = r[idx1] + r[idx2];
        dvec p1 = dvec(x[idx1], y[idx1], 0.0);
        dvec p2 = dvec(x[idx2], y[idx2], 0.0);
#if (NUM_DIM == 3)
        p1.z = z[idx1];
        p2.z = z[idx2];
#endif
        dvec distances = wrappedDifference(p1, p2, interval);
        const double distance = distances.getSquaredLength();
        if (radii * radii >= distance) {
            distances =
                distances * fZeroPerMuZero * (rsqrt(distance) - 1.0 / radii);
            atomicAdd(&vx[idx1], distances.x);
            atomicAdd(&vx[idx2], -distances.x);
            atomicAdd(&vy[idx1], distances.y);
            atomicAdd(&vy[idx2], -distances.y);
#if (NUM_DIM == 3)
            atomicAdd(&vz[idx1], distances.z);
            atomicAdd(&vz[idx2], -distances.z);
#endif
        }
    }
}

__global__ void velocityWallKernel(int numValues, double *r, double *x,
                                   double *y, double *z, double *vx, double *vy,
                                   double *vz, dvec lbb, dvec tfr,
                                   double fZeroPerMuZero, double dragCoeff) {
#if (PBC_X == 0 || PBC_Y == 0 || PBC_Z == 0)
    const int tid = getGlobalTid();
    if (tid < numValues) {
        double distance1 = 0.0;
        double distance2 = 0.0;
        double distance = 0.0;
        double xDrag = 1.0;
        double yDrag = 1.0;

#if (PBC_X == 0)
        distance1 = x[tid] - lbb.x;
        distance2 = x[tid] - tfr.x;
        distance = distance1 * distance1 < distance2 * distance2 ? distance1
                                                                 : distance2;
        if (r[tid] * r[tid] >= distance * distance) {
            const double direction = distance < 0 ? -1.0 : 1.0;
            distance *= direction;
            const double velocity = direction * distance * fZeroPerMuZero *
                                    (r[tid] - distance) / (r[tid] * distance);
            vx[tid] += velocity;
            xDrag = 1.0 - dragCoeff;

            // Drag of x wall to y & z
            vy[tid] *= xDrag;
            vz[tid] *= xDrag;
        }
#endif

#if (PBC_Y == 0)
        distance1 = y[tid] - lbb.y;
        distance2 = y[tid] - tfr.y;
        distance = distance1 * distance1 < distance2 * distance2 ? distance1
                                                                 : distance2;
        if (r[tid] * r[tid] >= distance * distance) {
            const double direction = distance < 0 ? -1.0 : 1.0;
            distance *= direction;
            const double velocity = direction * distance * fZeroPerMuZero *
                                    (r[tid] - distance) / (r[tid] * distance);

            // Retroactively apply possible drag from x wall to the velocity the
            // y wall causes
            vy[tid] += velocity * xDrag;
            yDrag = 1.0 - dragCoeff;

            // Drag of y wall to x & z
            vx[tid] *= yDrag;
            vz[tid] *= yDrag;
        }
#endif

#if (PBC_Z == 0)
        distance1 = z[tid] - lbb.z;
        distance2 = z[tid] - tfr.z;
        distance = distance1 * distance1 < distance2 * distance2 ? distance1
                                                                 : distance2;
        if (r[tid] * r[tid] >= distance * distance) {
            const double direction = distance < 0 ? -1.0 : 1.0;
            distance *= direction;
            const double velocity = direction * distance * fZeroPerMuZero *
                                    (r[tid] - distance) / (r[tid] * distance);

            // Retroactively apply possible drag from x & y walls to the
            // velocity the z wall causes
            vz[tid] += velocity * xDrag * yDrag;

            // Drag of z wall to x & y directions
            vx[tid] *= 1.0 - dragCoeff;
            vy[tid] *= 1.0 - dragCoeff;
        }
#endif
    }
#else
    return;
#endif
}

__global__ void neighborVelocityKernel(int *first, int *second,
                                       int *numNeighbors, double *sumX,
                                       double *sumY, double *sumZ, double *vx,
                                       double *vy, double *vz) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs;
         i += gridDim.x * blockDim.x) {
        const int idx1 = first[i];
        const int idx2 = second[i];
        atomicAdd(&numNeighbors[idx1], 1);
        atomicAdd(&numNeighbors[idx2], 1);

        atomicAdd(&sumX[idx1], vx[idx2]);
        atomicAdd(&sumX[idx2], vx[idx1]);

        atomicAdd(&sumY[idx1], vy[idx2]);
        atomicAdd(&sumY[idx2], vy[idx1]);

#if (NUM_DIM == 3)
        atomicAdd(&sumZ[idx1], vz[idx2]);
        atomicAdd(&sumZ[idx2], vz[idx1]);
#endif
    }
}

__global__ void flowVelocityKernel(int numValues, int *numNeighbors,
                                   double *velX, double *velY, double *velZ,
                                   double *nVelX, double *nVelY, double *nVelZ,
                                   double *posX, double *posY, double *posZ,
                                   double *r, dvec flowVel, dvec flowTfr,
                                   dvec flowLbb) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues;
         i += gridDim.x * blockDim.x) {
        const double multiplier =
            (numNeighbors[i] > 0 ? 1.0 / numNeighbors[i] : 0.0);

        int inside = (int)((posX[i] < flowTfr.x && posX[i] > flowLbb.x) ||
                           ((flowLbb.x - posX[i]) * (flowLbb.x - posX[i]) <=
                            r[i] * r[i]) ||
                           ((flowTfr.x - posX[i]) * (flowTfr.x - posX[i]) <=
                            r[i] * r[i]));

        inside *= (int)((posY[i] < flowTfr.y && posY[i] > flowLbb.y) ||
                        ((flowLbb.y - posY[i]) * (flowLbb.y - posY[i]) <=
                         r[i] * r[i]) ||
                        ((flowTfr.y - posY[i]) * (flowTfr.y - posY[i]) <=
                         r[i] * r[i]));

#if (NUM_DIM == 3)
        inside *= (int)((posZ[i] < flowTfr.z && posZ[i] > flowLbb.z) ||
                        ((flowLbb.z - posZ[i]) * (flowLbb.z - posZ[i]) <=
                         r[i] * r[i]) ||
                        ((flowTfr.z - posZ[i]) * (flowTfr.z - posZ[i]) <=
                         r[i] * r[i]));

        velZ[i] += !inside * multiplier * nVelZ[i] + flowVel.z * inside;
#endif

        velX[i] += !inside * multiplier * nVelX[i] + flowVel.x * inside;
        velY[i] += !inside * multiplier * nVelY[i] + flowVel.y * inside;
    }
}

__global__ void potentialEnergyKernel(int numValues, int *first, int *second,
                                      double *r, double *energy, dvec interval,
                                      double *x, double *y, double *z) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs;
         i += gridDim.x * blockDim.x) {
        const int idx1 = first[i];
        const int idx2 = second[i];
        dvec p1 = dvec(x[idx1], y[idx1], 0.0);
        dvec p2 = dvec(x[idx2], y[idx2], 0.0);
#if (NUM_DIM == 3)
        p1.z = z[idx1];
        p2.z = z[idx2];
#endif
        double e =
            r[idx1] + r[idx2] - wrappedDifference(p1, p2, interval).getLength();
        if (e > 0) {
            e *= e;
            atomicAdd(&energy[idx1], e);
            atomicAdd(&energy[idx2], e);
        }
    }
}

__global__ void gasExchangeKernel(int numValues, int *pair1, int *pair2,
                                  dvec interval, double *r, double *drdt,
                                  double *freeArea, double *x, double *y,
                                  double *z) {
    __shared__ double totalArea[128];
    __shared__ double totalOverlapArea[128];
    __shared__ double totalAreaPerRadius[128];
    __shared__ double totalOverlapAreaPerRadius[128];

    totalArea[threadIdx.x] = 0.0;
    totalOverlapArea[threadIdx.x] = 0.0;
    totalAreaPerRadius[threadIdx.x] = 0.0;
    totalOverlapAreaPerRadius[threadIdx.x] = 0.0;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs;
         i += gridDim.x * blockDim.x) {
        int idx1 = pair1[i];
        int idx2 = pair2[i];
        double r1 = r[idx1];
        double r2 = r[idx2];
        dvec p1 = dvec(x[idx1], y[idx1], 0.0);
        dvec p2 = dvec(x[idx2], y[idx2], 0.0);
#if (NUM_DIM == 3)
        p1.z = z[idx1];
        p2.z = z[idx2];
#endif
        double magnitude =
            wrappedDifference(p1, p2, interval).getSquaredLength();
        if (magnitude < (r1 + r2) * (r1 + r2)) {
            double overlapArea = 0;
            if (magnitude < r1 * r1 || magnitude < r2 * r2) {
                overlapArea = r1 < r2 ? r1 : r2;
                overlapArea *= overlapArea;
            } else {
                overlapArea =
                    0.5 * (r2 * r2 - r1 * r1 + magnitude) * rsqrt(magnitude);
                overlapArea *= overlapArea;
                overlapArea = r2 * r2 - overlapArea;
                overlapArea = overlapArea < 0 ? -overlapArea : overlapArea;
            }
#if (NUM_DIM == 3)
            overlapArea *= CUBBLE_PI;
#else
            overlapArea = 2.0 * sqrt(overlapArea);
#endif
            atomicAdd(&freeArea[idx1], overlapArea);
            atomicAdd(&freeArea[idx2], overlapArea);

            totalOverlapArea[threadIdx.x] += 2.0 * overlapArea;
            totalOverlapAreaPerRadius[threadIdx.x] +=
                overlapArea / r1 + overlapArea / r2;

            overlapArea *= (1.0 / r2 - 1.0 / r1);

            atomicAdd(&drdt[idx1], overlapArea);
            atomicAdd(&drdt[idx2], -overlapArea);
        }

        if (i < numValues) {
            double area = 2.0 * CUBBLE_PI * r[i];
#if (NUM_DIM == 3)
            area *= 2.0 * r[i];
#endif
            totalArea[threadIdx.x] += area;
            totalAreaPerRadius[threadIdx.x] += area / r[i];
        }
    }

    __syncthreads();

    if (threadIdx.x < 32) {
        totalArea[threadIdx.x] += totalArea[32 + threadIdx.x];
        totalArea[threadIdx.x] += totalArea[64 + threadIdx.x];
        totalArea[threadIdx.x] += totalArea[96 + threadIdx.x];

        totalOverlapArea[threadIdx.x] += totalOverlapArea[32 + threadIdx.x];
        totalOverlapArea[threadIdx.x] += totalOverlapArea[64 + threadIdx.x];
        totalOverlapArea[threadIdx.x] += totalOverlapArea[96 + threadIdx.x];

        totalAreaPerRadius[threadIdx.x] += totalAreaPerRadius[32 + threadIdx.x];
        totalAreaPerRadius[threadIdx.x] += totalAreaPerRadius[64 + threadIdx.x];
        totalAreaPerRadius[threadIdx.x] += totalAreaPerRadius[96 + threadIdx.x];

        totalOverlapAreaPerRadius[threadIdx.x] +=
            totalOverlapAreaPerRadius[32 + threadIdx.x];
        totalOverlapAreaPerRadius[threadIdx.x] +=
            totalOverlapAreaPerRadius[64 + threadIdx.x];
        totalOverlapAreaPerRadius[threadIdx.x] +=
            totalOverlapAreaPerRadius[96 + threadIdx.x];
    }

    if (threadIdx.x < 8) {
        totalArea[threadIdx.x] += totalArea[8 + threadIdx.x];
        totalArea[threadIdx.x] += totalArea[16 + threadIdx.x];
        totalArea[threadIdx.x] += totalArea[24 + threadIdx.x];

        totalOverlapArea[threadIdx.x] += totalOverlapArea[8 + threadIdx.x];
        totalOverlapArea[threadIdx.x] += totalOverlapArea[16 + threadIdx.x];
        totalOverlapArea[threadIdx.x] += totalOverlapArea[24 + threadIdx.x];

        totalAreaPerRadius[threadIdx.x] += totalAreaPerRadius[8 + threadIdx.x];
        totalAreaPerRadius[threadIdx.x] += totalAreaPerRadius[16 + threadIdx.x];
        totalAreaPerRadius[threadIdx.x] += totalAreaPerRadius[24 + threadIdx.x];

        totalOverlapAreaPerRadius[threadIdx.x] +=
            totalOverlapAreaPerRadius[8 + threadIdx.x];
        totalOverlapAreaPerRadius[threadIdx.x] +=
            totalOverlapAreaPerRadius[16 + threadIdx.x];
        totalOverlapAreaPerRadius[threadIdx.x] +=
            totalOverlapAreaPerRadius[24 + threadIdx.x];
    }

    if (threadIdx.x < 2) {
        totalArea[threadIdx.x] += totalArea[2 + threadIdx.x];
        totalArea[threadIdx.x] += totalArea[4 + threadIdx.x];
        totalArea[threadIdx.x] += totalArea[6 + threadIdx.x];

        totalOverlapArea[threadIdx.x] += totalOverlapArea[2 + threadIdx.x];
        totalOverlapArea[threadIdx.x] += totalOverlapArea[4 + threadIdx.x];
        totalOverlapArea[threadIdx.x] += totalOverlapArea[6 + threadIdx.x];

        totalAreaPerRadius[threadIdx.x] += totalAreaPerRadius[2 + threadIdx.x];
        totalAreaPerRadius[threadIdx.x] += totalAreaPerRadius[4 + threadIdx.x];
        totalAreaPerRadius[threadIdx.x] += totalAreaPerRadius[6 + threadIdx.x];

        totalOverlapAreaPerRadius[threadIdx.x] +=
            totalOverlapAreaPerRadius[2 + threadIdx.x];
        totalOverlapAreaPerRadius[threadIdx.x] +=
            totalOverlapAreaPerRadius[4 + threadIdx.x];
        totalOverlapAreaPerRadius[threadIdx.x] +=
            totalOverlapAreaPerRadius[6 + threadIdx.x];
    }

    if (threadIdx.x == 0) {
        totalArea[threadIdx.x] += totalArea[1];
        totalOverlapArea[threadIdx.x] += totalOverlapArea[1];
        totalAreaPerRadius[threadIdx.x] += totalAreaPerRadius[1];
        totalOverlapAreaPerRadius[threadIdx.x] += totalOverlapAreaPerRadius[1];

        atomicAdd(&dTotalArea, totalArea[0]);
        atomicAdd(&dTotalOverlapArea, totalOverlapArea[0]);
        atomicAdd(&dTotalAreaPerRadius, totalAreaPerRadius[0]);
        atomicAdd(&dTotalOverlapAreaPerRadius, totalOverlapAreaPerRadius[0]);
    }
}

__global__ void finalRadiusChangeRateKernel(double *drdt, double *r,
                                            double *freeArea, int numValues,
                                            double kappa, double kParam,
                                            double averageSurfaceAreaIn) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues;
         i += gridDim.x * blockDim.x) {
        double invRho = (dTotalAreaPerRadius - dTotalOverlapAreaPerRadius) /
                        (dTotalArea - dTotalOverlapArea);
        double area = 2.0 * CUBBLE_PI * r[i];
#if (NUM_DIM == 3)
        area *= 2.0 * r[i];
#endif
        const double vr = drdt[i] + kappa * averageSurfaceAreaIn * numValues /
                                        dTotalArea * (area - freeArea[i]) *
                                        (invRho - 1.0 / r[i]);
        drdt[i] = kParam * vr / area;
    }
}

__global__ void addVolume(double *r, int numValues) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues;
         i += gridDim.x * blockDim.x) {
        double multiplier = dVolumeMultiplier / dTotalVolume;
        multiplier += 1.0;

#if (NUM_DIM == 3)
        multiplier = cbrt(multiplier);
#else
        multiplier = sqrt(multiplier);
#endif
        r[i] = r[i] * multiplier;
    }
}

__global__ void calculateRedistributedGasVolume(double *volume, double *r,
                                                int *aboveMinRadFlags,
                                                int numValues) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numValues;
         i += gridDim.x * blockDim.x) {
        const double radius = r[i];
        double vol = CUBBLE_PI * radius * radius;
#if (NUM_DIM == 3)
        vol *= 1.333333333333333333333333 * radius;
#endif

        if (aboveMinRadFlags[i] == 0) {
            atomicAdd(&dVolumeMultiplier, vol);
            volume[i] = 0;
        } else
            volume[i] = vol;
    }
}

__device__ void adamsBashforth(int idx, double timeStep, double *yNext,
                               double *y, double *f, double *fPrevious) {
    yNext[idx] = y[idx] + 0.5 * timeStep * (3.0 * f[idx] - fPrevious[idx]);
}

__device__ double adamsMoulton(int idx, double timeStep, double *yNext,
                               double *y, double *f, double *fNext) {
    const double corrected = y[idx] + 0.5 * timeStep * (f[idx] + fNext[idx]);
    const double error = corrected - yNext[idx];
    yNext[idx] = corrected;

    return error < 0 ? -error : error;
}

__global__ void correctKernel(int numValues, double timeStep,
                              bool useGasExchange, double minRad,
                              double *errors, double *maxR, double *xp,
                              double *x, double *vx, double *vxp, double *yp,
                              double *y, double *vy, double *vyp, double *zp,
                              double *z, double *vz, double *vzp, double *rp,
                              double *r, double *vr, double *vrp) {
    int tid = threadIdx.x;
    __shared__ double me[128];
    __shared__ double mr[128];
    __shared__ int namr[128];
    me[tid] = 0.0;
    mr[tid] = 0.0;
    namr[tid] = 0;

    for (int i = tid + blockIdx.x * blockDim.x; i < numValues;
         i += blockDim.x * gridDim.x) {
        double corrected = x[i] + 0.5 * timeStep * (vx[i] + vxp[i]);
        double ex = corrected > xp[i] ? corrected - xp[i] : xp[i] - corrected;
        xp[i] = corrected;

        corrected = y[i] + 0.5 * timeStep * (vy[i] + vyp[i]);
        double ey = corrected > yp[i] ? corrected - yp[i] : yp[i] - corrected;
        yp[i] = corrected;

        double ez = 0.0;
#if (NUM_DIM == 3)
        corrected = z[i] + 0.5 * timeStep * (vz[i] + vzp[i]);
        ez = corrected > zp[i] ? corrected - zp[i] : z[i] - corrected;
        zp[i] = corrected;
#endif

        double er = 0.0;
        if (useGasExchange) {
            corrected = r[i] + 0.5 * timeStep * (vr[i] + vrp[i]);
            er = corrected > rp[i] ? corrected - rp[i] : rp[i] - corrected;
            rp[i] = corrected;
            mr[tid] = mr[tid] > corrected ? mr[tid] : corrected;
            namr[tid] += (int)(corrected > minRad);
        }
        corrected = ex > ey
                        ? (ex > ez ? (ex > er ? ex : er) : (ez > er ? ez : er))
                        : (ey > ez ? (ey > er ? ey : er) : (ez > er ? ez : er));
        me[tid] = me[tid] > corrected ? me[tid] : corrected;
    }

    __syncthreads();

    if (tid < 32) {
        me[tid] = me[tid] > me[32 + tid] ? me[tid] : me[32 + tid];
        me[tid] = me[tid] > me[64 + tid] ? me[tid] : me[64 + tid];
        me[tid] = me[tid] > me[96 + tid] ? me[tid] : me[96 + tid];

        mr[tid] = mr[tid] > mr[32 + tid] ? mr[tid] : mr[32 + tid];
        mr[tid] = mr[tid] > mr[64 + tid] ? mr[tid] : mr[64 + tid];
        mr[tid] = mr[tid] > mr[96 + tid] ? mr[tid] : mr[96 + tid];

        namr[tid] += namr[32 + tid];
        namr[tid] += namr[64 + tid];
        namr[tid] += namr[96 + tid];
    }

    if (tid < 8) {
        me[tid] = me[tid] > me[8 + tid] ? me[tid] : me[8 + tid];
        me[tid] = me[tid] > me[16 + tid] ? me[tid] : me[16 + tid];
        me[tid] = me[tid] > me[24 + tid] ? me[tid] : me[24 + tid];

        mr[tid] = mr[tid] > mr[8 + tid] ? mr[tid] : mr[8 + tid];
        mr[tid] = mr[tid] > mr[16 + tid] ? mr[tid] : mr[16 + tid];
        mr[tid] = mr[tid] > mr[24 + tid] ? mr[tid] : mr[24 + tid];

        namr[tid] += namr[8 + tid];
        namr[tid] += namr[16 + tid];
        namr[tid] += namr[24 + tid];
    }

    if (tid < 2) {
        me[tid] = me[tid] > me[2 + tid] ? me[tid] : me[2 + tid];
        me[tid] = me[tid] > me[4 + tid] ? me[tid] : me[4 + tid];
        me[tid] = me[tid] > me[6 + tid] ? me[tid] : me[6 + tid];

        mr[tid] = mr[tid] > mr[2 + tid] ? mr[tid] : mr[2 + tid];
        mr[tid] = mr[tid] > mr[4 + tid] ? mr[tid] : mr[4 + tid];
        mr[tid] = mr[tid] > mr[6 + tid] ? mr[tid] : mr[6 + tid];

        namr[tid] += namr[2 + tid];
        namr[tid] += namr[4 + tid];
        namr[tid] += namr[6 + tid];
    }

    if (tid == 0) {
        me[tid] = me[tid] > me[1] ? me[tid] : me[1];
        errors[blockIdx.x] = me[tid];

        mr[tid] = mr[tid] > mr[1] ? mr[tid] : mr[1];
        maxR[blockIdx.x] = mr[1];

        namr[tid] += namr[1];
        atomicAdd(&dNumBubblesAboveMinRad, namr[tid]);
    }
}

__global__ void miscEndStepKernel(int numValues, double *errors, double *maxR,
                                  int origBlockSize) {
    __shared__ double me[128];
    me[threadIdx.x] = 0.0;

    if (blockIdx.x < 2) {
        int tid = threadIdx.x;
        double *arr = nullptr;
        if (blockIdx.x == 0)
            arr = errors;
        if (blockIdx.x == 1)
            arr = maxR;

        for (int i = tid; i < origBlockSize; i += blockDim.x)
            me[tid] = me[tid] > arr[i] ? me[tid] : arr[i];

        __syncthreads();

        if (tid < 32) {
            me[tid] = me[tid] > me[32 + tid] ? me[tid] : me[32 + tid];
            me[tid] = me[tid] > me[64 + tid] ? me[tid] : me[64 + tid];
            me[tid] = me[tid] > me[96 + tid] ? me[tid] : me[96 + tid];
        }

        if (tid < 8) {
            me[tid] = me[tid] > me[8 + tid] ? me[tid] : me[8 + tid];
            me[tid] = me[tid] > me[16 + tid] ? me[tid] : me[16 + tid];
            me[tid] = me[tid] > me[24 + tid] ? me[tid] : me[24 + tid];
        }

        if (tid < 2) {
            me[tid] = me[tid] > me[2 + tid] ? me[tid] : me[2 + tid];
            me[tid] = me[tid] > me[4 + tid] ? me[tid] : me[4 + tid];
            me[tid] = me[tid] > me[6 + tid] ? me[tid] : me[6 + tid];
        }

        if (tid == 0)
            errors[blockIdx.x] = me[tid];
    }
}

__device__ void eulerIntegrate(int idx, double timeStep, double *y, double *f) {
    y[idx] += f[idx] * timeStep;
}

__device__ double calculateDistanceFromStart(int idx, double *x, double *xPrev,
                                             double *xStart,
                                             int *wrapMultiplier,
                                             double interval) {
    double distance = x[idx] - xStart[idx] + wrapMultiplier[idx] * interval;
    return distance * distance;
}

__device__ double calculatePathLength(int idx, double *x, double *xPrev,
                                      double *xStart, int *wrapMultiplier,
                                      double interval) {
    // Only works if done before boundary wrap
    const double diff = x[idx] - xPrev[idx];
    return diff * diff;
}

} // namespace cubble
