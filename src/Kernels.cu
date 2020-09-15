#include "Kernels.cuh"

namespace cubble {
__device__ Constants *dConstants;
__device__ double dTotalArea;
__device__ double dTotalOverlapArea;
__device__ double dTotalOverlapAreaPerRadius;
__device__ double dTotalAreaPerRadius;
__device__ double dTotalVolumeNew;
__device__ bool dErrorEncountered;
__device__ int dNumPairs;
__device__ int dNumPairsNew;
__device__ int dNumToBeDeleted;

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

__global__ void transformPositionsKernel(bool normalize, Bubbles bubbles) {
    const dvec lbb = dConstants->lbb;
    const dvec interval = dConstants->interval;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += gridDim.x * blockDim.x) {
        if (normalize) {
            bubbles.x[i] = (bubbles.x[i] - lbb.x) / interval.x;
            bubbles.y[i] = (bubbles.y[i] - lbb.y) / interval.y;
#if (NUM_DIM == 3)
            bubbles.z[i] = (bubbles.z[i] - lbb.z) / interval.z;
#endif
        } else {
            bubbles.x[i] = interval.x * bubbles.x[i] + lbb.x;
            bubbles.y[i] = interval.y * bubbles.y[i] + lbb.y;
#if (NUM_DIM == 3)
            bubbles.z[i] = interval.z * bubbles.z[i] + lbb.z;
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

__device__ int getCellIdxFromPos(double x, double y, double z, ivec cellDim) {
    const dvec lbb = dConstants->lbb;
    const dvec interval = dConstants->interval;
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
    // return idxVec.z * cellDim.x * cellDim.y + idxVec.y * cellDim.x +
    // idxVec.x;

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

__device__ void comparePair(int idx1, int idx2, Bubbles &bubbles,
                            Pairs &pairs) {
    const double maxDistance =
        bubbles.r[idx1] + bubbles.r[idx2] + dConstants->skinRadius;
    dvec p1 = dvec(bubbles.x[idx1], bubbles.y[idx1], 0.0);
    dvec p2 = dvec(bubbles.x[idx2], bubbles.y[idx2], 0.0);
#if (NUM_DIM == 3)
    p1.z = bubbles.z[idx1];
    p2.z = bubbles.z[idx2];
#endif
    if (wrappedDifference(p1, p2, dConstants->interval).getSquaredLength() <
        maxDistance * maxDistance) {
        // Set the smaller idx to idx1 and larger to idx2
        int id = idx1 > idx2 ? idx1 : idx2;
        idx1 = idx1 < idx2 ? idx1 : idx2;
        idx2 = id;

        atomicAdd(&bubbles.num_neighbors[idx1], 1);
        atomicAdd(&bubbles.num_neighbors[idx2], 1);
        id = atomicAdd(&dNumPairs, 1);
        pairs.i_copy[id] = idx1;
        pairs.j_copy[id] = idx2;
    }
}

__global__ void wrapKernel(Bubbles bubbles) {
    auto wrap = [](double *p, int *wc, double x, double low, double high,
                   int i) {
        int mult = x < low ? 1 : (x > high ? -1 : 0);
        p[i] = x + (high - low) * (double)mult;
        wc[i] -= mult;
    };

#if (PBC_X == 1 || PBC_Y == 1 || PBC_Z == 1)
    const dvec lbb = dConstants->lbb;
    const dvec tfr = dConstants->tfr;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += gridDim.x * blockDim.x) {
#if (PBC_X == 1)
        wrap(bubbles.x, bubbles.wrap_count_x, bubbles.x[i], lbb.x, tfr.x, i);
#endif
#if (PBC_Y == 1)
        wrap(bubbles.y, bubbles.wrap_count_y, bubbles.y[i], lbb.y, tfr.y, i);
#endif
#if (PBC_Z == 1)
        wrap(bubbles.z, bubbles.wrap_count_z, bubbles.z[i], lbb.z, tfr.z, i);
#endif
    }
#endif
}

__global__ void calculateVolumes(Bubbles bubbles) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += gridDim.x * blockDim.x) {
        const double radius = bubbles.r[i];
        double volume = radius * radius * CUBBLE_PI;
#if (NUM_DIM == 3)
        volume *= radius * 1.33333333333333333333333333;
#endif

        bubbles.temp_doubles[i] = volume;
    }
}

__global__ void assignDataToBubbles(ivec bubblesPerDim, double avgRad,
                                    Bubbles bubbles) {
    const dvec interval = dConstants->interval;
    const dvec lbb = dConstants->lbb;
    const dvec tfr = dConstants->tfr;
    const double minRad = dConstants->minRad;
    double *w = bubbles.rp;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += gridDim.x * blockDim.x) {
        bubbles.index[i] = i;
        dvec pos(0, 0, 0);
        pos.x = (i % bubblesPerDim.x) / (double)bubblesPerDim.x;
        pos.y =
            ((i / bubblesPerDim.x) % bubblesPerDim.y) / (double)bubblesPerDim.y;

        dvec randomOffset(bubbles.x[i], bubbles.y[i], 0);
#if (NUM_DIM == 3)
        randomOffset.z = bubbles.z[i];
        pos.z =
            (i / (bubblesPerDim.x * bubblesPerDim.y)) / (double)bubblesPerDim.z;
#endif
        pos *= interval;
        randomOffset = dvec::normalize(randomOffset) * avgRad * w[i];
        pos += randomOffset;

        double rad = bubbles.r[i];
        rad = abs(rad);
        rad = rad > minRad ? rad : rad + minRad;
        bubbles.r[i] = rad;
        bubbles.x[i] = pos.x > lbb.x
                           ? (pos.x < tfr.x ? pos.x : pos.x - interval.x)
                           : pos.x + interval.x;
        bubbles.y[i] = pos.y > lbb.y
                           ? (pos.y < tfr.y ? pos.y : pos.y - interval.y)
                           : pos.y + interval.y;
        bubbles.z[i] = pos.z > lbb.z
                           ? (pos.z < tfr.z ? pos.z : pos.z - interval.z)
                           : pos.z + interval.z;

        double area = 2.0 * CUBBLE_PI * rad;
#if (NUM_DIM == 3)
        area *= 2.0 * rad;
#endif
        w[i] = area / bubbles.count;
    }
}

__global__ void assignBubblesToCells(int *cellIndices, int *bubbleIndices,
                                     ivec cellDim, Bubbles bubbles) {
    const dvec lbb = dConstants->lbb;
    const dvec tfr = dConstants->tfr;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += gridDim.x * blockDim.x) {
        cellIndices[i] = getCellIdxFromPos(bubbles.x[i], bubbles.y[i],
                                           bubbles.z[i], cellDim);
        bubbleIndices[i] = i;
    }
}

__global__ void neighborSearch(int neighborCellNumber, int numCells,
                               int *offsets, int *sizes, Bubbles bubbles,
                               Pairs pairs) {
    const dvec interval = dConstants->interval;
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

                DEVICE_ASSERT(idx1 < bubbles.count, "Invalid bubble index!");
                DEVICE_ASSERT(idx2 < bubbles.count, "Invalid bubble index!");
                DEVICE_ASSERT(idx1 != idx2, "Invalid bubble index!");

                comparePair(idx1, idx2, bubbles, pairs);
                DEVICE_ASSERT(pairs.stride > dNumPairs,
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

                DEVICE_ASSERT(idx1 < bubbles.count, "Invalid bubble index!");
                DEVICE_ASSERT(idx2 < bubbles.count, "Invalid bubble index!");
                DEVICE_ASSERT(idx1 != idx2, "Invalid bubble index!");

                comparePair(idx1, idx2, bubbles, pairs);
                DEVICE_ASSERT(pairs.stride > dNumPairs,
                              "Too many neighbor indices!");
            }
        }
    }
}

__global__ void velocityPairKernel(Bubbles bubbles, Pairs pairs) {
    const dvec interval = dConstants->interval;
    const double fZeroPerMuZero = dConstants->fZeroPerMuZero;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs;
         i += gridDim.x * blockDim.x) {
        int idx1 = pairs.i[i];
        int idx2 = pairs.j[i];
        double radii = bubbles.rp[idx1] + bubbles.rp[idx2];
        dvec p1 = dvec(bubbles.xp[idx1], bubbles.yp[idx1], 0.0);
        dvec p2 = dvec(bubbles.xp[idx2], bubbles.yp[idx2], 0.0);
#if (NUM_DIM == 3)
        p1.z = bubbles.zp[idx1];
        p2.z = bubbles.zp[idx2];
#endif
        dvec distances = wrappedDifference(p1, p2, interval);
        const double distance = distances.getSquaredLength();
        if (radii * radii >= distance) {
            distances =
                distances * fZeroPerMuZero * (rsqrt(distance) - 1.0 / radii);
            atomicAdd(&bubbles.dxdtp[idx1], distances.x);
            atomicAdd(&bubbles.dxdtp[idx2], -distances.x);
            atomicAdd(&bubbles.dydtp[idx1], distances.y);
            atomicAdd(&bubbles.dydtp[idx2], -distances.y);
#if (NUM_DIM == 3)
            atomicAdd(&bubbles.dzdtp[idx1], distances.z);
            atomicAdd(&bubbles.dzdtp[idx2], -distances.z);
#endif
        }
    }
}

__global__ void velocityWallKernel(Bubbles bubbles) {
#if (PBC_X == 0 || PBC_Y == 0 || PBC_Z == 0)
    double distance1 = 0.0;
    double distance2 = 0.0;
    double distance = 0.0;
    double xDrag = 1.0;
    double yDrag = 1.0;
    const double drag = 1.0 - dConstants->wallDragStrength;
    const dvec lbb = dConstants->lbb;
    const dvec tfr = dConstants->tfr;
    const double fZeroPerMuZero = dConstants->fZeroPerMuZero;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += gridDim.x * blockDim.x) {
        const double rad = bubbles.rp[i];
#if (PBC_X == 0)
        const double xi = bubbles.xp[i];
        distance1 = xi - lbb.x;
        distance2 = xi - tfr.x;
        distance = distance1 * distance1 < distance2 * distance2 ? distance1
                                                                 : distance2;
        if (rad * rad >= distance * distance) {
            const double direction = distance < 0 ? -1.0 : 1.0;
            distance *= direction;
            const double velocity =
                direction * fZeroPerMuZero * (1.0 - distance / rad);
            bubbles.dxdtp[i] += velocity;
            xDrag = drag;

            // Drag of x wall to y & z
            bubbles.dydtp[i] *= drag;
            bubbles.dzdtp[i] *= drag;
        }
#endif

#if (PBC_Y == 0)
        const double yi = bubbles.yp[i];
        distance1 = yi - lbb.y;
        distance2 = yi - tfr.y;
        distance = distance1 * distance1 < distance2 * distance2 ? distance1
                                                                 : distance2;
        if (rad * rad >= distance * distance) {
            const double direction = distance < 0 ? -1.0 : 1.0;
            distance *= direction;
            const double velocity =
                direction * fZeroPerMuZero * (1.0 - distance / rad);

            // Retroactively apply possible drag from x wall to the velocity the
            // y wall causes
            bubbles.dydtp[i] += velocity * xDrag;
            yDrag = drag;

            // Drag of y wall to x & z
            bubbles.dxdtp[i] *= drag;
            bubbles.dzdtp[i] *= drag;
        }
#endif

#if (PBC_Z == 0)
        const double zi = bubbles.zp[i];
        distance1 = zi - lbb.z;
        distance2 = zi - tfr.z;
        distance = distance1 * distance1 < distance2 * distance2 ? distance1
                                                                 : distance2;
        if (rad * rad >= distance * distance) {
            const double direction = distance < 0 ? -1.0 : 1.0;
            distance *= direction;
            const double velocity =
                direction * fZeroPerMuZero * (1.0 - distance / rad);

            // Retroactively apply possible drag from x & y walls to the
            // velocity the z wall causes
            bubbles.dzdtp[i] += velocity * xDrag * yDrag;

            // Drag of z wall to x & y directions
            bubbles.dxdtp[i] *= drag;
            bubbles.dydtp[i] *= drag;
        }
#endif
    }
#endif
}

__global__ void neighborVelocityKernel(Bubbles bubbles, Pairs pairs) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs;
         i += gridDim.x * blockDim.x) {
        const int idx1 = pairs.i[i];
        const int idx2 = pairs.j[i];

        atomicAdd(&bubbles.flow_vx[idx1], bubbles.dxdto[idx2]);
        atomicAdd(&bubbles.flow_vx[idx2], bubbles.dxdto[idx1]);

        atomicAdd(&bubbles.flow_vy[idx1], bubbles.dydto[idx2]);
        atomicAdd(&bubbles.flow_vy[idx2], bubbles.dydto[idx1]);

#if (NUM_DIM == 3)
        atomicAdd(&bubbles.flow_vz[idx1], bubbles.dzdto[idx2]);
        atomicAdd(&bubbles.flow_vz[idx2], bubbles.dzdto[idx1]);
#endif
    }
}

__global__ void flowVelocityKernel(Bubbles bubbles) {
    const dvec flowVel = dConstants->flowVel;
    const dvec flowTfr = dConstants->flowTfr;
    const dvec flowLbb = dConstants->flowLbb;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += gridDim.x * blockDim.x) {
        double multiplier = bubbles.num_neighbors[i];
        multiplier = (multiplier > 0 ? 1.0 / multiplier : 0.0);
        const double xi = bubbles.xp[i];
        const double yi = bubbles.yp[i];
        double riSq = bubbles.rp[i];
        riSq *= riSq;

        int inside = (int)((xi < flowTfr.x && xi > flowLbb.x) ||
                           ((flowLbb.x - xi) * (flowLbb.x - xi) <= riSq) ||
                           ((flowTfr.x - xi) * (flowTfr.x - xi) <= riSq));

        inside *= (int)((yi < flowTfr.y && yi > flowLbb.y) ||
                        ((flowLbb.y - yi) * (flowLbb.y - yi) <= riSq) ||
                        ((flowTfr.y - yi) * (flowTfr.y - yi) <= riSq));

#if (NUM_DIM == 3)
        const double zi = bubbles.zp[i];
        inside *= (int)((zi < flowTfr.z && zi > flowLbb.z) ||
                        ((flowLbb.z - zi) * (flowLbb.z - zi) <= riSq) ||
                        ((flowTfr.z - zi) * (flowTfr.z - zi) <= riSq));

        bubbles.dzdtp[i] +=
            !inside * multiplier * bubbles.flow_vz[i] + flowVel.z * inside;
#endif

        // Either add the average velocity of neighbors or the imposed flow,
        // if the bubble is inside the flow area
        bubbles.dxdtp[i] +=
            !inside * multiplier * bubbles.flow_vx[i] + flowVel.x * inside;
        bubbles.dydtp[i] +=
            !inside * multiplier * bubbles.flow_vy[i] + flowVel.y * inside;
    }
}

__global__ void potentialEnergyKernel(Bubbles bubbles, Pairs pairs) {
    const dvec interval = dConstants->interval;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs;
         i += gridDim.x * blockDim.x) {
        const int idx1 = pairs.i[i];
        const int idx2 = pairs.j[i];
        dvec p1 = dvec(bubbles.x[idx1], bubbles.y[idx1], 0.0);
        dvec p2 = dvec(bubbles.x[idx2], bubbles.y[idx2], 0.0);
#if (NUM_DIM == 3)
        p1.z = bubbles.z[idx1];
        p2.z = bubbles.z[idx2];
#endif
        double e = bubbles.r[idx1] + bubbles.r[idx2] -
                   wrappedDifference(p1, p2, interval).getLength();
        if (e > 0) {
            e *= e;
            atomicAdd(&bubbles.temp_doubles[idx1], e);
            atomicAdd(&bubbles.temp_doubles[idx2], e);
        }
    }
}

__global__ void gasExchangeKernel(Bubbles bubbles, Pairs pairs) {
    const dvec interval = dConstants->interval;
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
        int idx1 = pairs.i[i];
        int idx2 = pairs.j[i];

        double r1 = bubbles.rp[idx1];
        double r2 = bubbles.rp[idx2];
        const double r1sq = r1 * r1;
        const double r2sq = r2 * r2;
        const double radii = r1 + r2;

        dvec p1 = dvec(bubbles.xp[idx1], bubbles.yp[idx1], 0.0);
        dvec p2 = dvec(bubbles.xp[idx2], bubbles.yp[idx2], 0.0);
#if (NUM_DIM == 3)
        p1.z = bubbles.zp[idx1];
        p2.z = bubbles.zp[idx2];
#endif
        double magnitude =
            wrappedDifference(p1, p2, interval).getSquaredLength();
        if (magnitude < radii * radii) {
            double overlapArea = 0;
            if (magnitude < r1sq || magnitude < r2sq) {
                overlapArea = r1sq < r2sq ? r1sq : r2sq;
            } else {
                overlapArea = 0.5 * (r2sq - r1sq + magnitude);
                overlapArea *= overlapArea;
                overlapArea /= magnitude;
                overlapArea = r2sq - overlapArea;
                overlapArea = overlapArea < 0 ? -overlapArea : overlapArea;
            }
#if (NUM_DIM == 3)
            overlapArea *= CUBBLE_PI;
#else
            overlapArea = 2.0 * sqrt(overlapArea);
#endif
            atomicAdd(&bubbles.temp_doubles[idx1], overlapArea);
            atomicAdd(&bubbles.temp_doubles[idx2], overlapArea);

            r1 = 1.0 / r1;
            r2 = 1.0 / r2;

            totalOverlapArea[threadIdx.x] += 2.0 * overlapArea;
            totalOverlapAreaPerRadius[threadIdx.x] += overlapArea * (r1 + r2);

            overlapArea *= (r2 - r1);

            atomicAdd(&bubbles.drdtp[idx1], overlapArea);
            atomicAdd(&bubbles.drdtp[idx2], -overlapArea);
        }

        if (i < bubbles.count) {
            r1 = bubbles.rp[i];
            double areaPerRad = 2.0 * CUBBLE_PI;
#if (NUM_DIM == 3)
            areaPerRad *= 2.0 * r1;
#endif
            totalArea[threadIdx.x] += areaPerRad * r1;
            totalAreaPerRadius[threadIdx.x] += areaPerRad;
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

        __syncwarp();

        if (threadIdx.x < 8) {
            totalArea[threadIdx.x] += totalArea[8 + threadIdx.x];
            totalArea[threadIdx.x] += totalArea[16 + threadIdx.x];
            totalArea[threadIdx.x] += totalArea[24 + threadIdx.x];

            totalOverlapArea[threadIdx.x] += totalOverlapArea[8 + threadIdx.x];
            totalOverlapArea[threadIdx.x] += totalOverlapArea[16 + threadIdx.x];
            totalOverlapArea[threadIdx.x] += totalOverlapArea[24 + threadIdx.x];

            totalAreaPerRadius[threadIdx.x] +=
                totalAreaPerRadius[8 + threadIdx.x];
            totalAreaPerRadius[threadIdx.x] +=
                totalAreaPerRadius[16 + threadIdx.x];
            totalAreaPerRadius[threadIdx.x] +=
                totalAreaPerRadius[24 + threadIdx.x];

            totalOverlapAreaPerRadius[threadIdx.x] +=
                totalOverlapAreaPerRadius[8 + threadIdx.x];
            totalOverlapAreaPerRadius[threadIdx.x] +=
                totalOverlapAreaPerRadius[16 + threadIdx.x];
            totalOverlapAreaPerRadius[threadIdx.x] +=
                totalOverlapAreaPerRadius[24 + threadIdx.x];

            __syncwarp();

            if (threadIdx.x < 2) {
                totalArea[threadIdx.x] += totalArea[2 + threadIdx.x];
                totalArea[threadIdx.x] += totalArea[4 + threadIdx.x];
                totalArea[threadIdx.x] += totalArea[6 + threadIdx.x];

                totalOverlapArea[threadIdx.x] +=
                    totalOverlapArea[2 + threadIdx.x];
                totalOverlapArea[threadIdx.x] +=
                    totalOverlapArea[4 + threadIdx.x];
                totalOverlapArea[threadIdx.x] +=
                    totalOverlapArea[6 + threadIdx.x];

                totalAreaPerRadius[threadIdx.x] +=
                    totalAreaPerRadius[2 + threadIdx.x];
                totalAreaPerRadius[threadIdx.x] +=
                    totalAreaPerRadius[4 + threadIdx.x];
                totalAreaPerRadius[threadIdx.x] +=
                    totalAreaPerRadius[6 + threadIdx.x];

                totalOverlapAreaPerRadius[threadIdx.x] +=
                    totalOverlapAreaPerRadius[2 + threadIdx.x];
                totalOverlapAreaPerRadius[threadIdx.x] +=
                    totalOverlapAreaPerRadius[4 + threadIdx.x];
                totalOverlapAreaPerRadius[threadIdx.x] +=
                    totalOverlapAreaPerRadius[6 + threadIdx.x];

                __syncwarp();

                if (threadIdx.x == 0) {
                    totalArea[threadIdx.x] += totalArea[1];
                    totalOverlapArea[threadIdx.x] += totalOverlapArea[1];
                    totalAreaPerRadius[threadIdx.x] += totalAreaPerRadius[1];
                    totalOverlapAreaPerRadius[threadIdx.x] +=
                        totalOverlapAreaPerRadius[1];

                    atomicAdd(&dTotalArea, totalArea[0]);
                    atomicAdd(&dTotalOverlapArea, totalOverlapArea[0]);
                    atomicAdd(&dTotalAreaPerRadius, totalAreaPerRadius[0]);
                    atomicAdd(&dTotalOverlapAreaPerRadius,
                              totalOverlapAreaPerRadius[0]);
                }
            }
        }
    }
}

__global__ void finalRadiusChangeRateKernel(Bubbles bubbles) {
    const double kappa = dConstants->kappa;
    const double kParameter = dConstants->kParameter;
    const double averageSurfaceAreaIn = dConstants->averageSurfaceAreaIn;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += gridDim.x * blockDim.x) {
        double invRho = (dTotalAreaPerRadius - dTotalOverlapAreaPerRadius) /
                        (dTotalArea - dTotalOverlapArea);
        const double rad = bubbles.rp[i];
        double area = 2.0 * CUBBLE_PI * rad;
#if (NUM_DIM == 3)
        area *= 2.0 * rad;
#endif
        const double vr = bubbles.drdtp[i] +
                          kappa * averageSurfaceAreaIn * bubbles.count /
                              dTotalArea * (area - bubbles.temp_doubles[i]) *
                              (invRho - 1.0 / rad);
        bubbles.drdtp[i] = kParameter * vr / area;
    }
}

__global__ void predictKernel(double timeStep, bool useGasExchange,
                              Bubbles bubbles) {
    // Adams-Bashforth integration
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += blockDim.x * gridDim.x) {
        bubbles.xp[i] =
            bubbles.x[i] +
            0.5 * timeStep * (3.0 * bubbles.dxdt[i] - bubbles.dxdto[i]);
        bubbles.yp[i] =
            bubbles.y[i] +
            0.5 * timeStep * (3.0 * bubbles.dydt[i] - bubbles.dydto[i]);
#if (NUM_DIM == 3)
        bubbles.zp[i] =
            bubbles.z[i] +
            0.5 * timeStep * (3.0 * bubbles.dzdt[i] - bubbles.dzdto[i]);
#endif
        if (useGasExchange) {
            bubbles.rp[i] =
                bubbles.r[i] +
                0.5 * timeStep * (3.0 * bubbles.drdt[i] - bubbles.drdto[i]);
        }
    }
}

__global__ void correctKernel(double timeStep, bool useGasExchange,
                              Bubbles bubbles) {
    const double minRad = dConstants->minRad;
    // Adams-Moulton integration
    int tid = threadIdx.x;
    // maximum error
    __shared__ double me[128];
    // maximum radius
    __shared__ double mr[128];
    // volume of remaining bubbles, i.e. new total volume
    __shared__ double tvn[128];
    // expansion, i.e. how far the boundary of a bubble has moved since
    // neighbors were last searched. Boundary expansion =
    // distance moved + radius increased with gas exchange.
    __shared__ double boundexp[128];
    me[tid] = 0.0;
    mr[tid] = 0.0;
    tvn[tid] = 0.0;
    boundexp[tid] = 0.0;

    for (int i = tid + blockIdx.x * blockDim.x; i < bubbles.count;
         i += blockDim.x * gridDim.x) {
        double predicted = bubbles.xp[i];
        double corrected =
            bubbles.x[i] +
            0.5 * timeStep * (bubbles.dxdt[i] + bubbles.dxdtp[i]);
        double ex = corrected > predicted ? corrected - predicted
                                          : predicted - corrected;
        bubbles.xp[i] = corrected;
        double delta = bubbles.saved_x[i] - corrected;
        double dist = delta * delta;

        predicted = bubbles.yp[i];
        corrected = bubbles.y[i] +
                    0.5 * timeStep * (bubbles.dydt[i] + bubbles.dydtp[i]);
        double ey = corrected > predicted ? corrected - predicted
                                          : predicted - corrected;
        bubbles.yp[i] = corrected;
        delta = bubbles.saved_y[i] - corrected;
        dist += delta * delta;

        double ez = 0.0;
#if (NUM_DIM == 3)
        predicted = bubbles.zp[i];
        corrected = bubbles.z[i] +
                    0.5 * timeStep * (bubbles.dzdt[i] + bubbles.dzdtp[i]);
        ez = corrected > predicted ? corrected - predicted
                                   : predicted - corrected;
        bubbles.zp[i] = corrected;
        delta = bubbles.saved_z[i] - corrected;
        dist += delta * delta;
#endif
        dist = sqrt(dist);

        double er = 0.0;
        if (useGasExchange) {
            predicted = bubbles.rp[i];
            corrected = bubbles.r[i] +
                        0.5 * timeStep * (bubbles.drdt[i] + bubbles.drdtp[i]);
            er = corrected > predicted ? corrected - predicted
                                       : predicted - corrected;
            bubbles.rp[i] = corrected;
            dist += corrected - bubbles.saved_r[i];

            double vol = corrected * corrected;
#if (NUM_DIM == 3)
            vol *= corrected;
#endif
            // Add remaining bubbles to new total volume
            if (corrected > minRad) {
                tvn[tid] += vol;
            } else {
                bubbles.temp_ints[atomicAdd(&dNumToBeDeleted, 1)] = i;
            }
            mr[tid] = mr[tid] > corrected ? mr[tid] : corrected;
        }
        corrected = ex > ey
                        ? (ex > ez ? (ex > er ? ex : er) : (ez > er ? ez : er))
                        : (ey > ez ? (ey > er ? ey : er) : (ez > er ? ez : er));
        // Store the maximum error per bubble in device memory
        // The errors are reset to zero between time steps
        ez = bubbles.error[i];
        bubbles.error[i] = corrected > ez ? corrected : ez;

        // Store the maximum error this thread has encountered
        me[tid] = me[tid] > corrected ? me[tid] : corrected;

        boundexp[tid] = boundexp[tid] > dist ? boundexp[tid] : dist;
    }

    __syncthreads();

    // Perform reductions for the values stored in shared memory
    if (tid < 32) {
        me[tid] = me[tid] > me[32 + tid] ? me[tid] : me[32 + tid];
        me[tid] = me[tid] > me[64 + tid] ? me[tid] : me[64 + tid];
        me[tid] = me[tid] > me[96 + tid] ? me[tid] : me[96 + tid];

        mr[tid] = mr[tid] > mr[32 + tid] ? mr[tid] : mr[32 + tid];
        mr[tid] = mr[tid] > mr[64 + tid] ? mr[tid] : mr[64 + tid];
        mr[tid] = mr[tid] > mr[96 + tid] ? mr[tid] : mr[96 + tid];

        boundexp[tid] = boundexp[tid] > boundexp[32 + tid] ? boundexp[tid]
                                                           : boundexp[32 + tid];
        boundexp[tid] = boundexp[tid] > boundexp[64 + tid] ? boundexp[tid]
                                                           : boundexp[64 + tid];
        boundexp[tid] = boundexp[tid] > boundexp[96 + tid] ? boundexp[tid]
                                                           : boundexp[96 + tid];

        tvn[tid] += tvn[32 + tid];
        tvn[tid] += tvn[64 + tid];
        tvn[tid] += tvn[96 + tid];
        __syncwarp();

        if (tid < 8) {
            me[tid] = me[tid] > me[8 + tid] ? me[tid] : me[8 + tid];
            me[tid] = me[tid] > me[16 + tid] ? me[tid] : me[16 + tid];
            me[tid] = me[tid] > me[24 + tid] ? me[tid] : me[24 + tid];

            mr[tid] = mr[tid] > mr[8 + tid] ? mr[tid] : mr[8 + tid];
            mr[tid] = mr[tid] > mr[16 + tid] ? mr[tid] : mr[16 + tid];
            mr[tid] = mr[tid] > mr[24 + tid] ? mr[tid] : mr[24 + tid];

            boundexp[tid] = boundexp[tid] > boundexp[8 + tid]
                                ? boundexp[tid]
                                : boundexp[8 + tid];
            boundexp[tid] = boundexp[tid] > boundexp[16 + tid]
                                ? boundexp[tid]
                                : boundexp[16 + tid];
            boundexp[tid] = boundexp[tid] > boundexp[24 + tid]
                                ? boundexp[tid]
                                : boundexp[24 + tid];

            tvn[tid] += tvn[8 + tid];
            tvn[tid] += tvn[16 + tid];
            tvn[tid] += tvn[24 + tid];
            __syncwarp();

            if (tid < 2) {
                me[tid] = me[tid] > me[2 + tid] ? me[tid] : me[2 + tid];
                me[tid] = me[tid] > me[4 + tid] ? me[tid] : me[4 + tid];
                me[tid] = me[tid] > me[6 + tid] ? me[tid] : me[6 + tid];

                mr[tid] = mr[tid] > mr[2 + tid] ? mr[tid] : mr[2 + tid];
                mr[tid] = mr[tid] > mr[4 + tid] ? mr[tid] : mr[4 + tid];
                mr[tid] = mr[tid] > mr[6 + tid] ? mr[tid] : mr[6 + tid];

                boundexp[tid] = boundexp[tid] > boundexp[2 + tid]
                                    ? boundexp[tid]
                                    : boundexp[2 + tid];
                boundexp[tid] = boundexp[tid] > boundexp[4 + tid]
                                    ? boundexp[tid]
                                    : boundexp[4 + tid];
                boundexp[tid] = boundexp[tid] > boundexp[6 + tid]
                                    ? boundexp[tid]
                                    : boundexp[6 + tid];

                tvn[tid] += tvn[2 + tid];
                tvn[tid] += tvn[4 + tid];
                tvn[tid] += tvn[6 + tid];
                __syncwarp();

                if (tid == 0) {
                    me[tid] = me[tid] > me[1] ? me[tid] : me[1];
                    bubbles.temp_doubles2[blockIdx.x] = me[tid];

                    mr[tid] = mr[tid] > mr[1] ? mr[tid] : mr[1];
                    bubbles.temp_doubles2[blockIdx.x + gridDim.x] = mr[tid];

                    boundexp[tid] = boundexp[tid] > boundexp[1] ? boundexp[tid]
                                                                : boundexp[1];
                    bubbles.temp_doubles2[blockIdx.x + 2 * gridDim.x] =
                        boundexp[tid];

                    tvn[tid] += tvn[1];
                    atomicAdd(&dTotalVolumeNew, tvn[tid]);
                }
            }
        }
    }
}

__global__ void endStepKernel(int origBlockSize, Bubbles bubbles) {
    __shared__ double me[128];
    me[threadIdx.x] = 0.0;

    if (blockIdx.x < 3) {
        int tid = threadIdx.x;
        double *arr = bubbles.temp_doubles2;
        if (blockIdx.x == 1)
            arr += origBlockSize;
        if (blockIdx.x == 2)
            arr += 2 * origBlockSize;

        for (int i = tid; i < origBlockSize; i += blockDim.x)
            me[tid] = me[tid] > arr[i] ? me[tid] : arr[i];

        __syncthreads();

        if (tid < 32) {
            me[tid] = me[tid] > me[32 + tid] ? me[tid] : me[32 + tid];
            me[tid] = me[tid] > me[64 + tid] ? me[tid] : me[64 + tid];
            me[tid] = me[tid] > me[96 + tid] ? me[tid] : me[96 + tid];

            __syncwarp();

            if (tid < 8) {
                me[tid] = me[tid] > me[8 + tid] ? me[tid] : me[8 + tid];
                me[tid] = me[tid] > me[16 + tid] ? me[tid] : me[16 + tid];
                me[tid] = me[tid] > me[24 + tid] ? me[tid] : me[24 + tid];

                __syncwarp();

                if (tid < 2) {
                    me[tid] = me[tid] > me[2 + tid] ? me[tid] : me[2 + tid];
                    me[tid] = me[tid] > me[4 + tid] ? me[tid] : me[4 + tid];
                    me[tid] = me[tid] > me[6 + tid] ? me[tid] : me[6 + tid];

                    __syncwarp();

                    if (tid == 0)
                        bubbles.temp_doubles2[blockIdx.x] = me[tid];
                }
            }
        }
    }
}

__global__ void eulerKernel(double timeStep, Bubbles bubbles) {
    // Euler integration
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += blockDim.x * gridDim.x) {
        bubbles.xp[i] += bubbles.dxdtp[i] * timeStep;
        bubbles.yp[i] += bubbles.dydtp[i] * timeStep;
#if (NUM_DIM == 3)
        bubbles.zp[i] += bubbles.dzdtp[i] * timeStep;
#endif
    }
}

__global__ void pathLengthDistanceKernel(Bubbles bubbles) {
    const dvec interval = dConstants->interval;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += blockDim.x * gridDim.x) {
        double diff = 0.0;
        double pl = 0.0;
        double fromStart = 0.0;
        double dist = 0.0;

        diff = bubbles.x[i] - bubbles.xp[i];
        pl += diff * diff;
        fromStart = bubbles.xp[i] - bubbles.x0[i] +
                    bubbles.wrap_count_x[i] * interval.x;
        dist += fromStart * fromStart;

        diff = bubbles.y[i] - bubbles.yp[i];
        pl += diff * diff;
        fromStart = bubbles.yp[i] - bubbles.y0[i] +
                    bubbles.wrap_count_y[i] * interval.y;
        dist += fromStart * fromStart;

#if (NUM_DIM == 3)
        diff = bubbles.z[i] - bubbles.zp[i];
        pl += diff * diff;
        fromStart = bubbles.zp[i] - bubbles.z0[i] +
                    bubbles.wrap_count_z[i] * interval.z;
        dist += fromStart * fromStart;
#endif
        bubbles.temp_doubles[i] = bubbles.path[i] + sqrt(pl);
        bubbles.distance[i] = dist;
    }
}

__global__ void swapDataCountPairs(Bubbles bubbles, Pairs pairs) {
    // Count of pairs to be deleted
    __shared__ int tbds[128];
    tbds[threadIdx.x] = 0;

    // The first 32 threads of the first block swap the data
    if (blockIdx.x == 0 && threadIdx.x < 32) {
        const int nNew = bubbles.count - dNumToBeDeleted;
        for (int i = threadIdx.x; i < dNumToBeDeleted; i += 32) {
            // If the to-be-deleted index is inside the remaining indices,
            // it will be swapped with one from the back that won't be
            // removed but which is outside the new range (i.e. would be
            // erroneously removed).
            const int idx1 = bubbles.temp_ints[i];
            if (idx1 < nNew) {
                // Count how many values before this ith value are swapped
                // from the back. In other words, count how many good values
                // to skip, before we choose which one to swap with idx1.
                int fromBack = i;
                int j = 0;
                while (j < i) {
                    if (bubbles.temp_ints[j] >= nNew) {
                        fromBack -= 1;
                    }
                    j += 1;
                }

                // Find the index idx2 of the jth remaining (i.e. "good")
                // value from the back, which still lies outside the new
                // range.
                j = 0;
                int idx2 = bubbles.count - 1;
                while (idx2 >= nNew && (bubbles.r[idx2] < dConstants->minRad ||
                                        j != fromBack)) {
                    if (bubbles.r[idx2] >= dConstants->minRad) {
                        j += 1;
                    }
                    idx2 -= 1;
                }

                // Append the old indices for later use
                bubbles.temp_ints[i + dNumToBeDeleted] = idx2;

                // Swap all the arrays
                swapValues(
                    idx2, idx1, bubbles.x, bubbles.y, bubbles.z, bubbles.r,
                    bubbles.dxdt, bubbles.dydt, bubbles.dzdt, bubbles.drdt,
                    bubbles.dxdto, bubbles.dydto, bubbles.dzdto, bubbles.drdto,
                    bubbles.x0, bubbles.y0, bubbles.z0, bubbles.saved_x,
                    bubbles.saved_y, bubbles.saved_z, bubbles.saved_r,
                    bubbles.path, bubbles.distance, bubbles.error,
                    bubbles.wrap_count_x, bubbles.wrap_count_y,
                    bubbles.wrap_count_z, bubbles.index, bubbles.num_neighbors);
            } else {
                bubbles.temp_ints[i + dNumToBeDeleted] = idx1;
            }
        }
    }

    // All threads check how many pairs are to be deleted
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < dNumPairs;
         i += blockDim.x * gridDim.x) {
        int j = 0;
        while (j < dNumToBeDeleted) {
            const int tbd = bubbles.temp_ints[j];
            if (pairs.i[i] == tbd || pairs.j[i] == tbd) {
                tbds[threadIdx.x] += 1;
            }
            j += 1;
        }
    }

    __syncthreads();

    if (threadIdx.x < 32) {
        tbds[threadIdx.x] += tbds[threadIdx.x + 32];
        tbds[threadIdx.x] += tbds[threadIdx.x + 64];
        tbds[threadIdx.x] += tbds[threadIdx.x + 96];

        __syncwarp();

        if (threadIdx.x < 8) {
            tbds[threadIdx.x] += tbds[threadIdx.x + 8];
            tbds[threadIdx.x] += tbds[threadIdx.x + 16];
            tbds[threadIdx.x] += tbds[threadIdx.x + 24];

            __syncwarp();

            if (threadIdx.x < 2) {
                tbds[threadIdx.x] += tbds[threadIdx.x + 2];
                tbds[threadIdx.x] += tbds[threadIdx.x + 4];
                tbds[threadIdx.x] += tbds[threadIdx.x + 6];

                __syncwarp();

                if (threadIdx.x < 1) {
                    tbds[threadIdx.x] += tbds[threadIdx.x + 1];
                    atomicAdd(&dNumPairsNew, -tbds[0]);
                }
            }
        }
    }
}

__global__ void addVolumeFixPairs(Bubbles bubbles, Pairs pairs) {
    double volMul = dTotalVolumeNew;
#if (NUM_DIM == 3)
    volMul = rcbrt(volMul);
#else
    volMul = rsqrt(volMul);
#endif
    volMul *= dConstants->bubbleVolumeMultiplier;

    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < dNumPairsNew;
         i += blockDim.x * gridDim.x) {
        // Check if either of the indices of this pair is any of the
        // to-be-deleted indices. If so, delete the pair, i.e.
        // swap a different pair on its place from the back of the list.
        int idx1 = pairs.i[i];
        int idx2 = pairs.j[i];
        int j = 0;
        while (j < dNumToBeDeleted) {
            int tbd = bubbles.temp_ints[j];
            if (idx1 == tbd || idx2 == tbd) {
                // Start from the back of pair list and go backwards until
                // neither of the pairs is in the to-be-deleted list.
                bool pairFound = false;
                int swapIdx = 0;
                while (!pairFound) {
                    pairFound = true;
                    swapIdx = atomicAdd(&dNumPairs, -1) - 1;
                    const int swap1 = pairs.i[swapIdx];
                    const int swap2 = pairs.j[swapIdx];
                    int k = 0;
                    while (k < dNumToBeDeleted) {
                        tbd = bubbles.temp_ints[k];
                        if (swap1 == tbd || swap2 == tbd) {
                            pairFound = false;
                            break;
                        }
                        k += 1;
                    }
                }
                pairs.i[i] = pairs.i[swapIdx];
                pairs.j[i] = pairs.j[swapIdx];
                break;
            }
            j += 1;
        }

        // Check if either of the indices should be updated. In other words,
        // if either of the indices is a value that is beyond the new range of
        // indices, change the value to the index it was swapped to by the
        // swapDataCountPairs kernel.
        idx1 = pairs.i[i];
        idx2 = pairs.j[i];
        j = 0;
        while (j < dNumToBeDeleted) {
            // The old, swapped indices were stored after the deleted indices
            int swapped = bubbles.temp_ints[dNumToBeDeleted + j];
            if (idx1 == swapped) {
                pairs.i[i] = bubbles.temp_ints[j];
            } else if (idx2 == swapped) {
                pairs.j[i] = bubbles.temp_ints[j];
            }

            j += 1;
        }

        if (i < bubbles.count - dNumToBeDeleted) {
            bubbles.r[i] = bubbles.r[i] * volMul;
        }
    }
}

__global__ void reorganizeByIndex(Bubbles bubbles, const int *newIndex) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += gridDim.x * blockDim.x) {
        int j = newIndex[i];
        // Predicteds become currents and vice versa,
        // saveds just store the currents until next search
        bubbles.xp[i] = bubbles.x[j];
        bubbles.yp[i] = bubbles.y[j];
        bubbles.zp[i] = bubbles.z[j];
        bubbles.rp[i] = bubbles.r[j];

        bubbles.saved_x[i] = bubbles.x[j];
        bubbles.saved_y[i] = bubbles.y[j];
        bubbles.saved_z[i] = bubbles.z[j];
        bubbles.saved_r[i] = bubbles.r[j];

        bubbles.dxdtp[i] = bubbles.dxdt[j];
        bubbles.dydtp[i] = bubbles.dydt[j];
        bubbles.dzdtp[i] = bubbles.dzdt[j];
        bubbles.drdtp[i] = bubbles.drdt[j];

        // Swap the rest in a 'loop' such that
        // flow_vx becomes dxdto becomes dydto
        // becomes dzdto becomes drdto becomes x0 etc.
        bubbles.flow_vx[i] = bubbles.dxdto[j];
        int k = newIndex[j];
        bubbles.dxdto[j] = bubbles.dydto[k];
        j = newIndex[k];
        bubbles.dydto[k] = bubbles.dzdto[j];
        k = newIndex[j];
        bubbles.dzdto[j] = bubbles.drdto[k];
        j = newIndex[k];
        bubbles.drdto[k] = bubbles.x0[j];
        k = newIndex[j];
        bubbles.x0[j] = bubbles.y0[k];
        j = newIndex[k];
        bubbles.y0[k] = bubbles.z0[j];
        k = newIndex[j];
        bubbles.z0[j] = bubbles.path[k];
        j = newIndex[k];
        bubbles.path[k] = bubbles.distance[j];
        k = newIndex[j];
        bubbles.distance[j] = bubbles.error[k];

        // Same loopy change for ints
        j = newIndex[i];
        bubbles.num_neighbors[i] = bubbles.wrap_count_x[j];
        k = newIndex[j];
        bubbles.wrap_count_x[j] = bubbles.wrap_count_y[k];
        j = newIndex[k];
        bubbles.wrap_count_y[k] = bubbles.wrap_count_z[j];
        k = newIndex[j];
        bubbles.wrap_count_z[j] = bubbles.index[k];

        // Additionally set the new num_neighbors to zero
        bubbles.index[k] = 0;
    }
}
} // namespace cubble
