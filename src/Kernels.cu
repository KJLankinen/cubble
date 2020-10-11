#include "Kernels.cuh"

namespace cubble {
__device__ Constants *dConstants;
__device__ double dTotalArea;
__device__ double dTotalOverlapArea;
__device__ double dTotalOverlapAreaPerRadius;
__device__ double dTotalAreaPerRadius;
__device__ double dTotalVolumeNew;
__device__ double dMaxError;
__device__ double dMaxRadius;
__device__ double dMaxExpansion;
__device__ bool dErrorEncountered;
__device__ int dNumPairs;
__device__ int dNumPairsNew;
__device__ int dNumToBeDeleted;
}; // namespace cubble

namespace cubble {
__global__ void cellByPosition(int *cellIndices, int *cellSizes, ivec cellDim,
                               Bubbles bubbles) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += gridDim.x * blockDim.x) {
        const int ci = getCellIdxFromPos(bubbles.x[i], bubbles.y[i],
                                         bubbles.z[i], cellDim);
        cellIndices[i] = ci;
        atomicAdd(&cellSizes[ci], 1);
    }
}

__global__ void indexByCell(int *cellIndices, int *cellOffsets,
                            int *bubbleIndices, int count) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < count;
         i += gridDim.x * blockDim.x) {
        bubbleIndices[atomicSub(&cellOffsets[cellIndices[i]], 1) - 1] = i;
    }
}

__device__ void comparePair(int idx1, int idx2, int *histogram, int *pairI,
                            int *pairJ, Bubbles &bubbles, Pairs &pairs) {
    const double maxDistance =
        bubbles.r[idx1] + bubbles.r[idx2] + dConstants->skinRadius;
    dvec p1 = dvec(bubbles.x[idx1], bubbles.y[idx1], 0.0);
    dvec p2 = dvec(bubbles.x[idx2], bubbles.y[idx2], 0.0);
    if (dConstants->dimensionality == 3) {
        p1.z = bubbles.z[idx1];
        p2.z = bubbles.z[idx2];
    }
    if (wrappedDifference(p1, p2, dConstants->interval).getSquaredLength() <
        maxDistance * maxDistance) {
        // Set the smaller idx to idx1 and larger to idx2
        int id = idx1 > idx2 ? idx1 : idx2;
        idx1 = idx1 < idx2 ? idx1 : idx2;
        idx2 = id;

        atomicAdd(&histogram[idx1], 1);
        id = atomicAdd(&dNumPairs, 1);
        pairI[id] = idx1;
        pairJ[id] = idx2;
    }
}

__global__ void neighborSearch(int numCells, int numNeighborCells, ivec cellDim,
                               int *offsets, int *sizes, int *histogram,
                               int *pairI, int *pairJ, Bubbles bubbles,
                               Pairs pairs) {
    DEVICE_ASSERT(blockDim.x >= 32, "Use at least 32 threads.");
    // Loop over each cell in the simulation box
    for (int i = blockIdx.x; i < numCells; i += gridDim.x) {
        const int s1 = sizes[i];
        if (0 == s1) {
            continue;
        }
        // Loop over each neighbor-cell-to-consider
        for (int j = threadIdx.x / 32; j < numNeighborCells;
             j += blockDim.x / 32) {
            const int ci = getNeighborCellIndex(i, cellDim, j);
            if (ci < 0) {
                continue;
            }
            const int s2 = sizes[ci];
            if (0 == s2) {
                continue;
            }

            const int o1 = offsets[i];
            const int o2 = offsets[ci];
            int numPairs = s1 * s2;
            if (ci == i) {
                // Comparing the cell to itself
                numPairs = (s1 * (s1 - 1)) / 2;
            }

            // Loop over each possible pair of bubbles in the cells
            for (int k = threadIdx.x % 32; k < numPairs; k += 32) {
                int b1 = 0;
                int b2 = 0;

                // If we're comparing a cell to itself, only compare the
                // "upper triangle" of values. Otherwise, compare all bubbles
                // of one cell to all bubbles of the other cell.
                // Insert the formula below to e.g. iPython to see what it
                // gives.
                if (ci == i) {
                    b1 =
                        s1 - 2 -
                        (int)floor(
                            sqrt(-8.0 * k + 4 * s1 * (s1 - 1) - 7) * 0.5 - 0.5);
                    b2 = o1 + k + b1 + 1 - s1 * (s1 - 1) / 2 +
                         (s1 - b1) * ((s1 - b1) - 1) / 2;
                    b1 += o1;
                } else {
                    b1 = o1 + k / s2;
                    b2 = o2 + k % s2;
                }

                DEVICE_ASSERT(b1 < bubbles.count, "Invalid bubble index!");
                DEVICE_ASSERT(b2 < bubbles.count, "Invalid bubble index!");
                DEVICE_ASSERT(b1 != b2, "Invalid bubble index!");

                comparePair(b1, b2, histogram, pairI, pairJ, bubbles, pairs);
                DEVICE_ASSERT(pairs.stride > dNumPairs,
                              "Too many neighbor indices!");
            }
        }
    }
}

__global__ void sortPairs(Bubbles bubbles, Pairs pairs, int *pairI,
                          int *pairJ) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs;
         i += gridDim.x * blockDim.x) {
        const int id = atomicSub(&bubbles.numNeighbors[pairI[i]], 1) - 1;
        pairs.i[id] = pairI[i];
        pairs.j[id] = pairJ[i];
    }
}

__global__ void countNumNeighbors(Bubbles bubbles, Pairs pairs) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs;
         i += gridDim.x * blockDim.x) {
        atomicAdd(&bubbles.numNeighbors[pairs.i[i]], 1);
        atomicAdd(&bubbles.numNeighbors[pairs.j[i]], 1);
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

        bubbles.savedX[i] = bubbles.x[j];
        bubbles.savedY[i] = bubbles.y[j];
        bubbles.savedZ[i] = bubbles.z[j];
        bubbles.savedR[i] = bubbles.r[j];

        bubbles.dxdtp[i] = bubbles.dxdt[j];
        bubbles.dydtp[i] = bubbles.dydt[j];
        bubbles.dzdtp[i] = bubbles.dzdt[j];
        bubbles.drdtp[i] = bubbles.drdt[j];

        // Swap the rest in a 'loop' such that
        // flow_vx becomes dxdto becomes dydto
        // becomes dzdto becomes drdto becomes x0 etc.
        bubbles.flowVx[i] = bubbles.dxdto[j];
        int k = newIndex[j];
        bubbles.dxdto[j] = bubbles.dydto[k];
        j = newIndex[k];
        bubbles.dydto[k] = bubbles.dzdto[j];
        k = newIndex[j];
        bubbles.dzdto[j] = bubbles.drdto[k];
        j = newIndex[k];
        bubbles.drdto[k] = bubbles.path[j];
        k = newIndex[j];
        bubbles.path[j] = bubbles.error[k];

        // Same loopy change for ints
        j = newIndex[i];
        bubbles.numNeighbors[i] = bubbles.wrapCountX[j];
        k = newIndex[j];
        bubbles.wrapCountX[j] = bubbles.wrapCountY[k];
        j = newIndex[k];
        bubbles.wrapCountY[k] = bubbles.wrapCountZ[j];
        k = newIndex[j];
        bubbles.wrapCountZ[j] = bubbles.index[k];

        // Additionally set the new num_neighbors to zero
        bubbles.index[k] = 0;
    }
}

__global__ void pairVelocity(Bubbles bubbles, Pairs pairs) {
    __shared__ double vx[BLOCK_SIZE];
    __shared__ double vy[BLOCK_SIZE];
    __shared__ double vz[BLOCK_SIZE];
    const dvec interval = dConstants->interval;
    const double fZeroPerMuZero = dConstants->fZeroPerMuZero;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs;
         i += gridDim.x * blockDim.x) {
        int idx1 = pairs.i[i];
        int idx2 = pairs.j[i];
        double radii = bubbles.rp[idx1] + bubbles.rp[idx2];
        dvec p1 = dvec(bubbles.xp[idx1], bubbles.yp[idx1], 0.0);
        dvec p2 = dvec(bubbles.xp[idx2], bubbles.yp[idx2], 0.0);
        if (dConstants->dimensionality == 3) {
            p1.z = bubbles.zp[idx1];
            p2.z = bubbles.zp[idx2];
        }
        dvec distances = wrappedDifference(p1, p2, interval);
        const double distance = distances.getSquaredLength();
        if (radii * radii >= distance) {
            distances =
                distances * fZeroPerMuZero * (rsqrt(distance) - 1.0 / radii);
            vx[threadIdx.x] = distances.x;
            vy[threadIdx.x] = distances.y;
            atomicAdd(&bubbles.dxdtp[idx2], -distances.x);
            atomicAdd(&bubbles.dydtp[idx2], -distances.y);
            if (dConstants->dimensionality == 3) {
                vz[threadIdx.x] = distances.z;
                atomicAdd(&bubbles.dzdtp[idx2], -distances.z);
            }
        } else {
            vx[threadIdx.x] = 0.0;
            vy[threadIdx.x] = 0.0;
            vz[threadIdx.x] = 0.0;
        }

        // pairs.i is an ordered list, such that the same index can be repeated
        // multiple times in a row. This means that many threads of a warp might
        // have the same address to save the values to. Here the threads choose
        // a leader amongst the threads with the same idx1, which sums the
        // values calculated by the warp and does a single atomicAdd per index.
        const unsigned int active = __activemask();
        __syncwarp(active);
        const unsigned int matches = __match_any_sync(active, idx1);
        const unsigned int lanemask_lt = (1 << (threadIdx.x & 31)) - 1;
        const unsigned int rank = __popc(matches & lanemask_lt);
        if (0 == rank) {
            double tx = 0.0;
            double ty = 0.0;
            double tz = 0.0;
            // thread id of the first lane of this warp, multiple of 32
            const int flt = 32 * (threadIdx.x >> 5);
#pragma unroll
            for (int j = 0; j < 32; j++) {
                const int mul = !!(matches & 1 << j);
                tx += vx[j + flt] * mul;
                ty += vy[j + flt] * mul;
                tz += vz[j + flt] * mul;
            }

            atomicAdd(&bubbles.dxdtp[idx1], tx);
            atomicAdd(&bubbles.dydtp[idx1], ty);
            if (dConstants->dimensionality == 3) {
                atomicAdd(&bubbles.dzdtp[idx1], tz);
            }
        }
        __syncwarp(active);
    }
}

__global__ void wallVelocity(Bubbles bubbles) {
    const double drag = 1.0 - dConstants->wallDragStrength;
    const dvec lbb = dConstants->lbb;
    const dvec tfr = dConstants->tfr;
    const double fZeroPerMuZero = dConstants->fZeroPerMuZero;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += gridDim.x * blockDim.x) {
        const double rad = bubbles.rp[i];
        double xDrag = 1.0;
        double yDrag = 1.0;
        double velocity = 0.0;

        auto touchesWall = [&i, &rad, &fZeroPerMuZero, &velocity](
                               double x, double low, double high) -> bool {
            double d1 = x - low;
            double d2 = x - high;
            d1 = d1 * d1 < d2 * d2 ? d1 : d2;
            if (rad * rad >= d1 * d1) {
                d2 = d1 < 0.0 ? -1.0 : 1.0;
                d1 *= d2;
                velocity = d2 * fZeroPerMuZero * (1.0 - d1 / rad);
                return true;
            }

            return false;
        };

        velocity = 0.0;
        if (dConstants->xWall && touchesWall(bubbles.xp[i], lbb.x, tfr.x)) {
            bubbles.dxdtp[i] += velocity;
            bubbles.dydtp[i] *= drag;
            bubbles.dzdtp[i] *= drag;
            xDrag = drag;
        }

        velocity = 0.0;
        if (dConstants->yWall && touchesWall(bubbles.yp[i], lbb.y, tfr.y)) {
            bubbles.dxdtp[i] *= drag;
            bubbles.dydtp[i] += velocity * xDrag;
            bubbles.dzdtp[i] *= drag;
            yDrag = drag;
        }

        velocity = 0.0;
        if (dConstants->zWall && touchesWall(bubbles.zp[i], lbb.z, tfr.z)) {
            bubbles.dxdtp[i] *= drag;
            bubbles.dydtp[i] *= drag;
            bubbles.dzdtp[i] += velocity * xDrag * yDrag;
        }
    }
}

__global__ void averageNeighborVelocity(Bubbles bubbles, Pairs pairs) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs;
         i += gridDim.x * blockDim.x) {
        const int idx1 = pairs.i[i];
        const int idx2 = pairs.j[i];

        atomicAdd(&bubbles.flowVx[idx1], bubbles.dxdto[idx2]);
        atomicAdd(&bubbles.flowVx[idx2], bubbles.dxdto[idx1]);

        atomicAdd(&bubbles.flowVy[idx1], bubbles.dydto[idx2]);
        atomicAdd(&bubbles.flowVy[idx2], bubbles.dydto[idx1]);

        if (dConstants->dimensionality == 3) {
            atomicAdd(&bubbles.flowVz[idx1], bubbles.dzdto[idx2]);
            atomicAdd(&bubbles.flowVz[idx2], bubbles.dzdto[idx1]);
        }
    }
}

__global__ void imposedFlowVelocity(Bubbles bubbles) {
    const dvec flowVel = dConstants->flowVel;
    const dvec flowTfr = dConstants->flowTfr;
    const dvec flowLbb = dConstants->flowLbb;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += gridDim.x * blockDim.x) {
        double multiplier = bubbles.numNeighbors[i];
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

        if (dConstants->dimensionality == 3) {
            const double zi = bubbles.zp[i];
            inside *= (int)((zi < flowTfr.z && zi > flowLbb.z) ||
                            ((flowLbb.z - zi) * (flowLbb.z - zi) <= riSq) ||
                            ((flowTfr.z - zi) * (flowTfr.z - zi) <= riSq));

            bubbles.dzdtp[i] +=
                !inside * multiplier * bubbles.flowVz[i] + flowVel.z * inside;
        }

        // Either add the average velocity of neighbors or the imposed flow,
        // if the bubble is inside the flow area
        bubbles.dxdtp[i] +=
            !inside * multiplier * bubbles.flowVx[i] + flowVel.x * inside;
        bubbles.dydtp[i] +=
            !inside * multiplier * bubbles.flowVy[i] + flowVel.y * inside;
    }
}

__global__ void potentialEnergy(Bubbles bubbles, Pairs pairs, double *energy) {
    const dvec interval = dConstants->interval;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs;
         i += gridDim.x * blockDim.x) {
        const int idx1 = pairs.i[i];
        const int idx2 = pairs.j[i];
        dvec p1 = dvec(bubbles.x[idx1], bubbles.y[idx1], 0.0);
        dvec p2 = dvec(bubbles.x[idx2], bubbles.y[idx2], 0.0);
        if (dConstants->dimensionality == 3) {
            p1.z = bubbles.z[idx1];
            p2.z = bubbles.z[idx2];
        }
        double e = bubbles.r[idx1] + bubbles.r[idx2] -
                   wrappedDifference(p1, p2, interval).getLength();
        if (e > 0) {
            e *= e;
            atomicAdd(&energy[idx1], e);
            atomicAdd(&energy[idx2], e);
        }
    }
}

__global__ void pairwiseGasExchange(Bubbles bubbles, Pairs pairs,
                                    double *overlap) {
    // Gas exchange between bubbles, a.k.a. local gas exchange
    const dvec interval = dConstants->interval;
    __shared__ double sbuf[2 * BLOCK_SIZE];
    __shared__ double ta[BLOCK_SIZE]       // total area of all bubbles
        __shared__ double toa[BLOCK_SIZE]; // total overlap area
    __shared__ double tapr[BLOCK_SIZE];    // ta per radius
    __shared__ double toapr[BLOCK_SIZE];   // toa per radius
    const int tid = threadIdx.x;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs;
         i += gridDim.x * blockDim.x) {
        int idx1 = pairs.i[i];
        int idx2 = pairs.j[i];

        DEVICE_ASSERT(idx1 != idx2, "Bubble is a pair with itself");

        if (i < bubbles.count) {
            double r1 = bubbles.rp[i];
            double areaPerRad = 2.0 * CUBBLE_PI;
            if (dConstants->dimensionality == 3) {
                areaPerRad *= 2.0 * r1;
            }
            ta[tid] += areaPerRad * r1;
            tapr[tid] += areaPerRad;
        }

        double r1 = bubbles.rp[idx1];
        double r2 = bubbles.rp[idx2];
        const double r1sq = r1 * r1;
        const double r2sq = r2 * r2;
        const double radii = r1 + r2;

        dvec p1 = dvec(bubbles.xp[idx1], bubbles.yp[idx1], 0.0);
        dvec p2 = dvec(bubbles.xp[idx2], bubbles.yp[idx2], 0.0);
        if (dConstants->dimensionality == 3) {
            p1.z = bubbles.zp[idx1];
            p2.z = bubbles.zp[idx2];
        }

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

            if (dConstants->dimensionality == 3) {
                overlapArea *= CUBBLE_PI;
            } else {
                overlapArea = 2.0 * sqrt(overlapArea);
            }

            sbuf[tid] = overlapArea;
            atomicAdd(&overlap[idx2], overlapArea);

            r1 = 1.0 / r1;
            r2 = 1.0 / r2;

            toa[tid] += 2.0 * overlapArea;
            toapr[tid] += overlapArea * (r1 + r2);

            overlapArea *= (r2 - r1);

            sbuf[tid + BLOCK_SIZE] = overlapArea;
            atomicAdd(&bubbles.drdtp[idx2], -overlapArea);
        } else {
            sbuf[tid] = 0.0;
            sbuf[tid + BLOCK_SIZE] = 0.0;
        }

        // pairs.i is an ordered list, such that the same index can be repeated
        // multiple times in a row. This means that many threads of a warp might
        // have the same address to save the values to. Here the threads choose
        // a leader amongst the threads with the same idx1, which sums the
        // values calculated by the warp and does a single atomicAdd per index.
        const unsigned int active = __activemask();
        __syncwarp(active);
        const unsigned int matches = __match_any_sync(active, idx1);
        const unsigned int lanemask_lt = (1 << (threadIdx.x & 31)) - 1;
        const unsigned int rank = __popc(matches & lanemask_lt);
        if (0 == rank) {
            double oa = 0.0;
            double vr = 0.0;
            // thread id of the first lane of this warp, multiple of 32
            const int flt = 32 * (threadIdx.x >> 5);
#pragma unroll
            for (int j = 0; j < 32; j++) {
                const int mul = !!(matches & 1 << j);
                oa += sbuf[j + flt] * mul;
                vr += sbuf[j + flt + BLOCK_SIZE] * mul;
            }

            atomicAdd(&overlap[idx1], oa);
            atomicAdd(&bubbles.drdtp[idx1], vr);
        }
        __syncwarp(active);
    }

    __syncthreads();

    if (tid < 32) {
#pragma unroll
        for (int i = 1; i < BLOCK_SIZE / 32; i++) {
            ta[tid] += ta[tid + i * 32];
            toa[tid] += toa[tid + i * 32];
            tapr[tid] += tapr[tid + i * 32];
            toapr[tid] += toapr[tid + i * 32];
        }
        __syncwarp();

        double temp[4];
        temp[0] = ta[tid ^ 16];
        temp[1] = toa[tid ^ 16];
        temp[2] = tapr[tid ^ 16];
        temp[3] = toapr[tid ^ 16];
        __syncwarp();
        ta[tid] += temp[0];
        toa[tid] += temp[1];
        tapr[tid] += temp[2];
        toapr[tid] += temp[3];
        __syncwarp();

        temp[0] = ta[tid ^ 8];
        temp[1] = toa[tid ^ 8];
        temp[2] = tapr[tid ^ 8];
        temp[3] = toapr[tid ^ 8];
        __syncwarp();
        ta[tid] += temp[0];
        toa[tid] += temp[1];
        tapr[tid] += temp[2];
        toapr[tid] += temp[3];
        __syncwarp();

        temp[0] = ta[tid ^ 4];
        temp[1] = toa[tid ^ 4];
        temp[2] = tapr[tid ^ 4];
        temp[3] = toapr[tid ^ 4];
        __syncwarp();
        ta[tid] += temp[0];
        toa[tid] += temp[1];
        tapr[tid] += temp[2];
        toapr[tid] += temp[3];
        __syncwarp();

        temp[0] = ta[tid ^ 2];
        temp[1] = toa[tid ^ 2];
        temp[2] = tapr[tid ^ 2];
        temp[3] = toapr[tid ^ 2];
        __syncwarp();
        ta[tid] += temp[0];
        toa[tid] += temp[1];
        tapr[tid] += temp[2];
        toapr[tid] += temp[3];
        __syncwarp();
    }

    if (0 == tid) {
        atomicAdd(&dTotalArea, ta[0] + ta[1]);
        atomicAdd(&dTotalAreaPerRadius, tapr[0] + tapr[1]);
        atomicAdd(&dTotalOverlapArea, toa[0] + toa[1]);
        atomicAdd(&dTotalOverlapAreaPerRadius, toapr[0] + toapr[1]);
    }
}

__global__ void mediatedGasExchange(Bubbles bubbles, double *overlap) {
    // Gas exchange mediated by the liquid surrounding the bubbles,
    // a.k.a. global gas exchange
    const double kappa = dConstants->kappa;
    const double kParameter = dConstants->kParameter;
    const double averageSurfaceAreaIn = dConstants->averageSurfaceAreaIn;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += gridDim.x * blockDim.x) {
        double invRho = (dTotalAreaPerRadius - dTotalOverlapAreaPerRadius) /
                        (dTotalArea - dTotalOverlapArea);
        const double rad = bubbles.rp[i];
        double area = 2.0 * CUBBLE_PI * rad;
        if (dConstants->dimensionality == 3) {
            area *= 2.0 * rad;
        }
        const double vr = bubbles.drdtp[i] + kappa * averageSurfaceAreaIn *
                                                 bubbles.count / dTotalArea *
                                                 (area - overlap[i]) *
                                                 (invRho - 1.0 / rad);
        bubbles.drdtp[i] = kParameter * vr / area;
    }
}

__global__ void predict(double timeStep, bool useGasExchange, Bubbles bubbles) {
    // Adams-Bashforth integration
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += blockDim.x * gridDim.x) {
        if (useGasExchange) {
            bubbles.rp[i] =
                bubbles.r[i] +
                0.5 * timeStep * (3.0 * bubbles.drdt[i] - bubbles.drdto[i]);
        } else {
            bubbles.xp[i] =
                bubbles.x[i] +
                0.5 * timeStep * (3.0 * bubbles.dxdt[i] - bubbles.dxdto[i]);
            bubbles.yp[i] =
                bubbles.y[i] +
                0.5 * timeStep * (3.0 * bubbles.dydt[i] - bubbles.dydto[i]);
            if (dConstants->dimensionality == 3) {
                bubbles.zp[i] =
                    bubbles.z[i] +
                    0.5 * timeStep * (3.0 * bubbles.dzdt[i] - bubbles.dzdto[i]);
            }
        }
    }
}

__global__ void correct(double timeStep, bool useGasExchange, Bubbles bubbles,
                        double *maximums, int *toBeDeleted) {
    // Adams-Moulton integration
    const double minRad = dConstants->minRad;
    int tid = threadIdx.x;
    // maximum error
    __shared__ double me[BLOCK_SIZE];
    // maximum radius
    __shared__ double mr[BLOCK_SIZE];
    // volume of remaining bubbles, i.e. new total volume
    __shared__ double tvn[BLOCK_SIZE];
    // boundary expansion, i.e. how far the boundary of a bubble has moved since
    // neighbors were last searched. Boundary expansion =
    // distance moved + radius increased with gas exchange.
    __shared__ double be[BLOCK_SIZE];
    me[tid] = 0.0;
    mr[tid] = 0.0;
    tvn[tid] = 0.0;
    be[tid] = 0.0;

    for (int i = tid + blockIdx.x * blockDim.x; i < bubbles.count;
         i += blockDim.x * gridDim.x) {
        double delta = 0.0;
        auto correctPrediction = [&timeStep, &i, &delta](double *p, double *pp,
                                                         double *po, double *v,
                                                         double *vp) -> double {
            double predicted = pp[i];
            double corrected = p[i] + 0.5 * timeStep * (v[i] + vp[i]);
            pp[i] = corrected;
            delta = corrected - po[i];

            predicted -= corrected;
            return predicted < 0.0 ? -predicted : predicted;
        };

        double maxErr = correctPrediction(bubbles.x, bubbles.xp, bubbles.savedX,
                                          bubbles.dxdt, bubbles.dxdtp);
        double dist = delta * delta;
        maxErr = fmax(maxErr,
                      correctPrediction(bubbles.y, bubbles.yp, bubbles.savedY,
                                        bubbles.dydt, bubbles.dydtp));
        dist += delta * delta;
        if (dConstants->dimensionality == 3) {
            maxErr = fmax(
                maxErr, correctPrediction(bubbles.z, bubbles.zp, bubbles.savedZ,
                                          bubbles.dzdt, bubbles.dzdtp));
            dist += delta * delta;
        }
        dist = sqrt(dist);

        if (useGasExchange) {
            maxErr = fmax(
                maxErr, correctPrediction(bubbles.r, bubbles.rp, bubbles.savedR,
                                          bubbles.drdt, bubbles.drdtp));
            dist += delta;
            // Predicted value has been overwritten by the corrected value
            // inside the lambda
            double rad = bubbles.rp[i];
            double vol = rad * rad;
            if (dConstants->dimensionality == 3) {
                vol *= rad;
            }
            // Add remaining bubbles to new total volume
            if (rad > minRad) {
                tvn[tid] += vol;
            } else {
                toBeDeleted[atomicAdd(&dNumToBeDeleted, 1)] = i;
            }
            mr[tid] = fmax(mr[tid], rad);
        }
        // Store the maximum error per bubble in device memory
        // The errors are reset to zero between time steps
        bubbles.error[i] = fmax(maxErr, bubbles.error[i]);

        me[tid] = fmax(me[tid], maxErr);
        be[tid] = fmax(be[tid], dist);
    }

    __syncthreads();

    auto reduceMax = [&tid](auto *arr, int base) {
        arr[tid] =
            fmax(fmax(fmax(arr[tid], arr[tid + base]), arr[tid + 2 * base]),
                 arr[tid + 3 * base]);
    };

    // Perform reductions for the values stored in shared memory
    if (tid < 32) {
        reduceMax(me, 32);
        reduceMax(mr, 32);
        reduceMax(be, 32);

        tvn[tid] += tvn[32 + tid] + tvn[64 + tid] + tvn[96 + tid];
        __syncwarp();

        if (tid < 8) {
            reduceMax(me, 8);
            reduceMax(mr, 8);
            reduceMax(be, 8);

            tvn[tid] += tvn[8 + tid] + tvn[16 + tid] + tvn[24 + tid];
            __syncwarp();

            if (tid < 2) {
                reduceMax(me, 2);
                reduceMax(mr, 2);
                reduceMax(be, 2);

                tvn[tid] += tvn[2 + tid] + tvn[4 + tid] + tvn[6 + tid];
                __syncwarp();

                if (tid == 0) {
                    maximums[blockIdx.x] = fmax(me[tid], me[1]);
                    maximums[blockIdx.x + gridDim.x] = fmax(mr[tid], mr[1]);
                    maximums[blockIdx.x + 2 * gridDim.x] = fmax(be[tid], be[1]);
                    atomicAdd(&dTotalVolumeNew, tvn[tid] + tvn[1]);
                }
            }
        }
    }
}

__global__ void incrementPath(Bubbles bubbles) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += blockDim.x * gridDim.x) {
        double diff = bubbles.x[i] - bubbles.xp[i];
        double pl = diff * diff;
        diff = bubbles.y[i] - bubbles.yp[i];
        pl += diff * diff;
        if (dConstants->dimensionality == 3) {
            diff = bubbles.z[i] - bubbles.zp[i];
            pl += diff * diff;
        }
        bubbles.path[i] += sqrt(pl);
    }
}

__global__ void swapDataCountPairs(Bubbles bubbles, Pairs pairs,
                                   int *toBeDeleted) {
    // Count of pairs to be deleted
    __shared__ int tbds[BLOCK_SIZE];
    int tid = threadIdx.x;
    tbds[tid] = 0;

    // The first 32 threads of the first block swap the data
    if (blockIdx.x == 0 && tid < 32) {
        const int nNew = bubbles.count - dNumToBeDeleted;
        for (int i = tid; i < dNumToBeDeleted; i += 32) {
            // If the to-be-deleted index is inside the remaining indices,
            // it will be swapped with one from the back that won't be
            // removed but which is outside the new range (i.e. would be
            // erroneously removed).
            const int idx1 = toBeDeleted[i];
            if (idx1 < nNew) {
                // Count how many values before this ith value are swapped
                // from the back. In other words, count how many good values
                // to skip, before we choose which one to swap with idx1.
                int fromBack = i;
                int j = 0;
                while (j < i) {
                    if (toBeDeleted[j] >= nNew) {
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
                toBeDeleted[i + dNumToBeDeleted] = idx2;

                // Swap all the arrays
                swapValues(idx2, idx1, bubbles.x, bubbles.y, bubbles.z,
                           bubbles.r, bubbles.dxdt, bubbles.dydt, bubbles.dzdt,
                           bubbles.drdt, bubbles.dxdto, bubbles.dydto,
                           bubbles.dzdto, bubbles.drdto, bubbles.savedX,
                           bubbles.savedY, bubbles.savedZ, bubbles.savedR,
                           bubbles.path, bubbles.error, bubbles.wrapCountX,
                           bubbles.wrapCountY, bubbles.wrapCountZ,
                           bubbles.index, bubbles.numNeighbors);
            } else {
                toBeDeleted[i + dNumToBeDeleted] = idx1;
            }
        }
    }

    // All threads check how many pairs are to be deleted
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < dNumPairs;
         i += blockDim.x * gridDim.x) {
        int j = 0;
        while (j < dNumToBeDeleted) {
            const int tbd = toBeDeleted[j];
            if (pairs.i[i] == tbd || pairs.j[i] == tbd) {
                tbds[tid] += 1;
            }
            j += 1;
        }
    }

    __syncthreads();

    auto reduceSum = [&tid](auto *arr, int base) {
        arr[tid] += arr[tid + base] + arr[tid + 2 * base] + arr[tid + 3 * base];
    };

    if (tid < 32) {
        reduceSum(tbds, 32);
        __syncwarp();

        if (tid < 8) {
            reduceSum(tbds, 8);
            __syncwarp();

            if (tid < 2) {
                reduceSum(tbds, 2);
                __syncwarp();

                if (tid < 1) {
                    tbds[tid] += tbds[tid + 1];
                    atomicAdd(&dNumPairsNew, -tbds[0]);
                }
            }
        }
    }
}

__global__ void addVolumeFixPairs(Bubbles bubbles, Pairs pairs,
                                  int *toBeDeleted) {
    double volMul = dTotalVolumeNew;
    if (dConstants->dimensionality == 3) {
        volMul = rcbrt(volMul);
    } else {
        volMul = rsqrt(volMul);
    }
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
            int tbd = toBeDeleted[j];
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
                        tbd = toBeDeleted[k];
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
            int swapped = toBeDeleted[dNumToBeDeleted + j];
            if (idx1 == swapped) {
                pairs.i[i] = toBeDeleted[j];
            } else if (idx2 == swapped) {
                pairs.j[i] = toBeDeleted[j];
            }

            j += 1;
        }

        if (i < bubbles.count - dNumToBeDeleted) {
            bubbles.r[i] = bubbles.r[i] * volMul;
        }
    }
}

__global__ void euler(double timeStep, Bubbles bubbles) {
    // Euler integration
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += blockDim.x * gridDim.x) {
        bubbles.xp[i] += bubbles.dxdtp[i] * timeStep;
        bubbles.yp[i] += bubbles.dydtp[i] * timeStep;
        if (dConstants->dimensionality == 3) {
            bubbles.zp[i] += bubbles.dzdtp[i] * timeStep;
        }
    }
}

__global__ void transformPositions(bool normalize, Bubbles bubbles) {
    const dvec lbb = dConstants->lbb;
    const dvec interval = dConstants->interval;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += gridDim.x * blockDim.x) {
        if (normalize) {
            bubbles.x[i] = (bubbles.x[i] - lbb.x) / interval.x;
            bubbles.y[i] = (bubbles.y[i] - lbb.y) / interval.y;
            if (dConstants->dimensionality == 3) {
                bubbles.z[i] = (bubbles.z[i] - lbb.z) / interval.z;
            }
        } else {
            bubbles.x[i] = interval.x * bubbles.x[i] + lbb.x;
            bubbles.y[i] = interval.y * bubbles.y[i] + lbb.y;
            if (dConstants->dimensionality == 3) {
                bubbles.z[i] = interval.z * bubbles.z[i] + lbb.z;
            }
        }
    }
}

__global__ void wrapOverPeriodicBoundaries(Bubbles bubbles) {
    const dvec lbb = dConstants->lbb;
    const dvec tfr = dConstants->tfr;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += gridDim.x * blockDim.x) {
        auto wrap = [&i](double *p, int *wc, double x, double low,
                         double high) {
            int mult = x < low ? 1 : (x > high ? -1 : 0);
            p[i] = x + (high - low) * (double)mult;
            wc[i] -= mult;
        };

        if (!dConstants->xWall)
            wrap(bubbles.x, bubbles.wrapCountX, bubbles.x[i], lbb.x, tfr.x);
        if (!dConstants->yWall)
            wrap(bubbles.y, bubbles.wrapCountY, bubbles.y[i], lbb.y, tfr.y);
        if (!dConstants->zWall)
            wrap(bubbles.z, bubbles.wrapCountZ, bubbles.z[i], lbb.z, tfr.z);
    }
}

__global__ void calculateVolumes(Bubbles bubbles, double *volumes) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += gridDim.x * blockDim.x) {
        const double radius = bubbles.r[i];
        double volume = radius * radius * CUBBLE_PI;
        if (dConstants->dimensionality == 3) {
            volume *= radius * 1.33333333333333333333333333;
        }

        volumes[i] = volume;
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
        if (dConstants->dimensionality == 3) {
            randomOffset.z = bubbles.z[i];
            pos.z = (i / (bubblesPerDim.x * bubblesPerDim.y)) /
                    (double)bubblesPerDim.z;
        }
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
        if (dConstants->dimensionality == 3) {
            area *= 2.0 * rad;
        }
        w[i] = area / bubbles.count;
    }
}

__global__ void initGlobals() {
    if (0 == threadIdx.x + blockIdx.x) {
        resetDeviceGlobals();
    }
}

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

__device__ dvec wrappedDifference(dvec p1, dvec p2, dvec interval) {
    const dvec d1 = p1 - p2;
    dvec d2 = d1;
    dvec temp = interval - d1.getAbsolute();
    if (!dConstants->xWall && temp.x * temp.x < d1.x * d1.x) {
        d2.x = temp.x * (d1.x < 0 ? 1.0 : -1.0);
    }
    if (!dConstants->yWall && temp.y * temp.y < d1.y * d1.y) {
        d2.y = temp.y * (d1.y < 0 ? 1.0 : -1.0);
    }
    if (dConstants->dimensionality == 3 && !dConstants->zWall &&
        temp.z * temp.z < d1.z * d1.z) {
        d2.z = temp.z * (d1.z < 0 ? 1.0 : -1.0);
    }

    return d2;
}

__device__ int getNeighborCellIndex(int cellIdx, ivec dim, int neighborNum) {
    ivec idxVec = get3DIdxFrom1DIdx(cellIdx, dim);
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

    if (!dConstants->xWall) {
        idxVec.x += dim.x;
        idxVec.x %= dim.x;
    } else if (idxVec.x < 0 || idxVec.x >= dim.x) {
        return -1;
    }

    if (!dConstants->yWall) {
        idxVec.y += dim.y;
        idxVec.y %= dim.y;
    } else if (idxVec.y < 0 || idxVec.y >= dim.y) {
        return -1;
    }

    if (!dConstants->zWall) {
        idxVec.z += dim.z;
        idxVec.z %= dim.z;
    } else if (idxVec.z < 0 || idxVec.z >= dim.z) {
        return -1;
    }

    return get1DIdxFrom3DIdx(idxVec, dim);
}

__device__ int getCellIdxFromPos(double x, double y, double z, ivec cellDim) {
    const dvec lbb = dConstants->lbb;
    const dvec interval = dConstants->interval;
    const int xid = floor(cellDim.x * (x - lbb.x) / interval.x);
    const int yid = floor(cellDim.y * (y - lbb.y) / interval.y);
    int zid = 0;
    if (dConstants->dimensionality == 3) {
        zid = floor(cellDim.z * (z - lbb.z) / interval.z);
    }

    return get1DIdxFrom3DIdx(ivec(xid, yid, zid), cellDim);
}

__device__ int get1DIdxFrom3DIdx(ivec idxVec, ivec cellDim) {
    // Linear encoding
    return idxVec.z * cellDim.x * cellDim.y + idxVec.y * cellDim.x + idxVec.x;
}

__device__ ivec get3DIdxFrom1DIdx(int idx, ivec cellDim) {
    ivec idxVec(0, 0, 0);
    // Linear decoding
    idxVec.x = idx % cellDim.x;
    idxVec.y = (idx / cellDim.x) % cellDim.y;
    if (dConstants->dimensionality == 3) {
        idxVec.z = idx / (cellDim.x * cellDim.y);
    }

    return idxVec;
}

__device__ void resetDeviceGlobals() {
    dTotalArea = 0.0;
    dTotalOverlapArea = 0.0;
    dTotalOverlapAreaPerRadius = 0.0;
    dTotalAreaPerRadius = 0.0;
    dTotalVolumeNew = 0.0;
    dMaxError = 0.0;
    dMaxRadius = 0.0;
    dMaxExpansion = 0.0;
    dNumToBeDeleted = 0;
    dNumPairsNew = dNumPairs;
    dErrorEncountered = false;
}
} // namespace cubble
