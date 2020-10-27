/*
    Cubble
    Copyright (C) 2019  Juhana Lankinen

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

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
    if (wrappedDifference(bubbles.x[idx1], bubbles.y[idx1], bubbles.z[idx1],
                          bubbles.x[idx2], bubbles.y[idx2], bubbles.z[idx2])
            .getSquaredLength() < maxDistance * maxDistance) {
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
    __shared__ double vx[BLOCK_SIZE];
    __shared__ double vy[BLOCK_SIZE];
    __shared__ double vz[BLOCK_SIZE];

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs;
         i += gridDim.x * blockDim.x) {
        const int idx1 = pairs.i[i];
        const int idx2 = pairs.j[i];

        vx[threadIdx.x] = bubbles.dxdto[idx2];
        vy[threadIdx.x] = bubbles.dydto[idx2];

        atomicAdd(&bubbles.flowVx[idx2], bubbles.dxdto[idx1]);
        atomicAdd(&bubbles.flowVy[idx2], bubbles.dydto[idx1]);

        if (dConstants->dimensionality == 3) {
            vz[threadIdx.x] = bubbles.dzdto[idx2];
            atomicAdd(&bubbles.flowVz[idx2], bubbles.dzdto[idx1]);
        }

        const unsigned int active = __activemask();
        __syncwarp(active);
        double tx = 0.0;
        double ty = 0.0;
        double tz = 0.0;
        if (0 == warpReduceMatching(active, idx1, &sum<double>, &tx, vx, &ty,
                                    vy, &tz, vz)) {
            atomicAdd(&bubbles.flowVx[idx1], tx);
            atomicAdd(&bubbles.flowVy[idx1], ty);
            if (dConstants->dimensionality == 3) {
                atomicAdd(&bubbles.flowVz[idx1], tz);
            }
        }
        __syncwarp(active);
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
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs;
         i += gridDim.x * blockDim.x) {
        const int idx1 = pairs.i[i];
        const int idx2 = pairs.j[i];
        double e =
            bubbles.r[idx1] + bubbles.r[idx2] -
            wrappedDifference(bubbles.x[idx1], bubbles.y[idx1], bubbles.z[idx1],
                              bubbles.x[idx2], bubbles.y[idx2], bubbles.z[idx2])
                .getLength();
        if (e > 0) {
            e *= e;
            atomicAdd(&energy[idx1], e);
            atomicAdd(&energy[idx2], e);
        }
    }
}

__global__ void pairwiseInteraction(Bubbles bubbles, Pairs pairs,
                                    double *overlap, bool useGasExchange) {
    __shared__ double sbuf[9 * BLOCK_SIZE];
    sbuf[threadIdx.x + 5 * BLOCK_SIZE] = 0.0;
    sbuf[threadIdx.x + 6 * BLOCK_SIZE] = 0.0;
    sbuf[threadIdx.x + 7 * BLOCK_SIZE] = 0.0;
    sbuf[threadIdx.x + 8 * BLOCK_SIZE] = 0.0;
    // If phi is very small, there could be zero pairs, but this loop is assumed
    // to run at least bubbles.count times for the per bubble sums
    const int n = dNumPairs > bubbles.count ? dNumPairs : bubbles.count;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
         i += gridDim.x * blockDim.x) {
        // Per bubble calculations
        if (useGasExchange && i < bubbles.count) {
            double r = bubbles.rp[i];
            double areaPerRad = 2.0 * CUBBLE_PI;
            if (dConstants->dimensionality == 3) {
                areaPerRad *= 2.0 * r;
            }
            sbuf[threadIdx.x + 5 * BLOCK_SIZE] += areaPerRad * r;
            sbuf[threadIdx.x + 6 * BLOCK_SIZE] += areaPerRad;
        }

        // Per pair calculations
        if (i < dNumPairs) {
            const int idx1 = pairs.i[i];
            const int idx2 = pairs.j[i];
            double r1 = bubbles.rp[idx1];
            double r2 = bubbles.rp[idx2];
            const double radii = r1 + r2;
            dvec distances = wrappedDifference(
                bubbles.xp[idx1], bubbles.yp[idx1], bubbles.zp[idx1],
                bubbles.xp[idx2], bubbles.yp[idx2], bubbles.zp[idx2]);
            double magnitude = distances.getSquaredLength();

            if (radii * radii >= magnitude) {
                // Pair velocities
                distances = distances * dConstants->fZeroPerMuZero *
                            (rsqrt(magnitude) - 1.0 / radii);
                sbuf[threadIdx.x + 0 * BLOCK_SIZE] = distances.x;
                sbuf[threadIdx.x + 1 * BLOCK_SIZE] = distances.y;
                atomicAdd(&bubbles.dxdtp[idx2], -distances.x);
                atomicAdd(&bubbles.dydtp[idx2], -distances.y);
                if (dConstants->dimensionality == 3) {
                    sbuf[threadIdx.x + 2 * BLOCK_SIZE] = distances.z;
                    atomicAdd(&bubbles.dzdtp[idx2], -distances.z);
                }

                // Pairwise gas exchange
                if (useGasExchange) {
                    const double r1sq = r1 * r1;
                    const double r2sq = r2 * r2;
                    double overlapArea = 0;
                    if (magnitude < r1sq || magnitude < r2sq) {
                        overlapArea = r1sq < r2sq ? r1sq : r2sq;
                    } else {
                        overlapArea = 0.5 * (r2sq - r1sq + magnitude);
                        overlapArea *= overlapArea;
                        overlapArea /= magnitude;
                        overlapArea = r2sq - overlapArea;
                        overlapArea =
                            overlapArea < 0 ? -overlapArea : overlapArea;
                    }

                    if (dConstants->dimensionality == 3) {
                        overlapArea *= CUBBLE_PI;
                    } else {
                        overlapArea = 2.0 * sqrt(overlapArea);
                    }

                    sbuf[threadIdx.x + 3 * BLOCK_SIZE] = overlapArea;
                    atomicAdd(&overlap[idx2], overlapArea);

                    r1 = 1.0 / r1;
                    r2 = 1.0 / r2;

                    sbuf[threadIdx.x + 7 * BLOCK_SIZE] += 2.0 * overlapArea;
                    sbuf[threadIdx.x + 8 * BLOCK_SIZE] +=
                        overlapArea * (r1 + r2);

                    overlapArea *= (r2 - r1);

                    sbuf[threadIdx.x + 4 * BLOCK_SIZE] = overlapArea;
                    atomicAdd(&bubbles.drdtp[idx2], -overlapArea);
                }
            } else {
                continue;
            }

            const unsigned int active = __activemask();
            __syncwarp(active);
            double oa = 0.0;
            double vr = 0.0;
            double vx = 0.0;
            double vy = 0.0;
            double vz = 0.0;
            if (0 == warpReduceMatching(
                         active, idx1, &sum<double>, &vx, &sbuf[0 * BLOCK_SIZE],
                         &vy, &sbuf[1 * BLOCK_SIZE], &vz, &sbuf[2 * BLOCK_SIZE],
                         &oa, &sbuf[3 * BLOCK_SIZE], &vr,
                         &sbuf[4 * BLOCK_SIZE])) {
                atomicAdd(&overlap[idx1], oa);
                atomicAdd(&bubbles.drdtp[idx1], vr);
                atomicAdd(&bubbles.dxdtp[idx1], vx);
                atomicAdd(&bubbles.dydtp[idx1], vy);
                if (dConstants->dimensionality == 3) {
                    atomicAdd(&bubbles.dzdtp[idx1], vz);
                }
            }
            __syncwarp(active);
        }
    }

    if (useGasExchange) {
        __syncthreads();
        const int warpNum = threadIdx.x >> 5;
        const int wid = threadIdx.x & 31;
        if (threadIdx.x < 32) {
            reduce(&sbuf[5 * BLOCK_SIZE], warpNum, &sum);
            if (0 == wid) {
                atomicAdd(&dTotalArea, sbuf[threadIdx.x + 5 * BLOCK_SIZE]);
            }
        } else if (threadIdx.x < 64) {
            reduce(&sbuf[6 * BLOCK_SIZE], warpNum, &sum);
            if (0 == wid) {
                atomicAdd(&dTotalAreaPerRadius,
                          sbuf[threadIdx.x + 6 * BLOCK_SIZE]);
            }
        } else if (threadIdx.x < 96) {
            reduce(&sbuf[7 * BLOCK_SIZE], warpNum, &sum);
            if (0 == wid) {
                atomicAdd(&dTotalOverlapArea,
                          sbuf[threadIdx.x + 7 * BLOCK_SIZE]);
            }
        } else if (threadIdx.x < 128) {
            reduce(&sbuf[8 * BLOCK_SIZE], warpNum, &sum);
            if (0 == wid) {
                atomicAdd(&dTotalOverlapAreaPerRadius,
                          sbuf[threadIdx.x + 8 * BLOCK_SIZE]);
            }
        }
    }
}

__global__ void preIntegrate(double ts, bool useGasExchange, Bubbles bubbles,
                             double *temp1, double *temp2) {
    // Adams-Bashforth integration
    if (threadIdx.x + blockIdx.x == 0) {
        resetDeviceGlobals();
    }

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += blockDim.x * gridDim.x) {
        temp1[i] = 0.0;
        temp2[i] = 0.0;

        bubbles.dxdtp[i] = 0.0;
        bubbles.xp[i] = bubbles.x[i] +
                        0.5 * ts * (3.0 * bubbles.dxdt[i] - bubbles.dxdto[i]);

        bubbles.dydtp[i] = 0.0;
        bubbles.yp[i] = bubbles.y[i] +
                        0.5 * ts * (3.0 * bubbles.dydt[i] - bubbles.dydto[i]);

        if (dConstants->dimensionality == 3) {
            bubbles.dzdtp[i] = 0.0;
            bubbles.zp[i] =
                bubbles.z[i] +
                0.5 * ts * (3.0 * bubbles.dzdt[i] - bubbles.dzdto[i]);
        }

        if (useGasExchange) {
            bubbles.drdtp[i] = 0.0;
            bubbles.rp[i] =
                bubbles.r[i] +
                0.5 * ts * (3.0 * bubbles.drdt[i] - bubbles.drdto[i]);
        }
    }
}

__global__ void postIntegrate(double ts, bool useGasExchange,
                              bool incrementPath, Bubbles bubbles,
                              double *maximums, double *overlap,
                              int *toBeDeleted) {
    // Adams-Moulton integration
    __shared__ double sbuf[4 * BLOCK_SIZE];
    sbuf[threadIdx.x + 0 * BLOCK_SIZE] = 0.0;
    sbuf[threadIdx.x + 1 * BLOCK_SIZE] = 0.0;
    sbuf[threadIdx.x + 2 * BLOCK_SIZE] = 0.0;
    sbuf[threadIdx.x + 3 * BLOCK_SIZE] = 0.0;

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += blockDim.x * gridDim.x) {
        double pred = 0.0;
        double corr = 0.0;
        double maxErr = 0.0;
        double dist = 0.0;

        // X
        pred = bubbles.xp[i];
        corr = bubbles.x[i] + 0.5 * ts * (bubbles.dxdt[i] + bubbles.dxdtp[i]);
        bubbles.xp[i] = corr;
        maxErr = fmax(abs(pred - corr), maxErr);
        pred = corr - bubbles.savedX[i];
        dist += pred * pred;

        // Y
        pred = bubbles.yp[i];
        corr = bubbles.y[i] + 0.5 * ts * (bubbles.dydt[i] + bubbles.dydtp[i]);
        bubbles.yp[i] = corr;
        maxErr = fmax(abs(pred - corr), maxErr);
        pred = corr - bubbles.savedY[i];
        dist += pred * pred;

        // Z
        if (3 == dConstants->dimensionality) {
            pred = bubbles.zp[i];
            corr =
                bubbles.z[i] + 0.5 * ts * (bubbles.dzdt[i] + bubbles.dzdtp[i]);
            bubbles.zp[i] = corr;
            maxErr = fmax(abs(pred - corr), maxErr);
            pred = corr - bubbles.savedZ[i];
            dist += pred * pred;
        }

        dist = sqrt(dist);

        // R
        if (useGasExchange) {
            // First update the drdtp by the liquid mediated gas exchange
            double invRho = (dTotalAreaPerRadius - dTotalOverlapAreaPerRadius) /
                            (dTotalArea - dTotalOverlapArea);
            pred = bubbles.rp[i];
            corr = 2.0 * CUBBLE_PI * pred;
            if (dConstants->dimensionality == 3) {
                corr *= 2.0 * pred;
            }

            double vr = bubbles.drdtp[i] +
                        dConstants->kappa * dConstants->averageSurfaceAreaIn *
                            bubbles.count / dTotalArea * (corr - overlap[i]) *
                            (invRho - 1.0 / pred);
            vr = dConstants->kParameter * vr / corr;
            bubbles.drdtp[i] = vr;

            // Correct
            corr = bubbles.r[i] + 0.5 * ts * (bubbles.drdt[i] + vr);
            bubbles.rp[i] = corr;
            maxErr = fmax(abs(pred - corr), maxErr);
            pred = corr - bubbles.savedR[i];
            dist += pred;

            // Calculate volume
            pred = corr * corr;
            if (dConstants->dimensionality == 3) {
                pred *= corr;
            }

            // Add remaining bubbles to new total volume
            if (corr > dConstants->minRad) {
                sbuf[threadIdx.x + 3 * BLOCK_SIZE] += pred;
            } else {
                toBeDeleted[atomicAdd(&dNumToBeDeleted, 1)] = i;
            }

            sbuf[threadIdx.x + 1 * BLOCK_SIZE] =
                fmax(sbuf[threadIdx.x + 1 * BLOCK_SIZE], corr);
        }

        // Path
        if (incrementPath) {
            pred = bubbles.x[i] - bubbles.xp[i];
            corr = pred * pred;

            pred = bubbles.y[i] - bubbles.yp[i];
            corr += pred * pred;

            if (dConstants->dimensionality == 3) {
                pred = bubbles.z[i] - bubbles.zp[i];
                corr += pred * pred;
            }

            overlap[i] = bubbles.path[i] + sqrt(corr);
        }

        // Store the maximum error per bubble in device memory
        // The errors are reset to zero between time steps
        bubbles.error[i] = fmax(maxErr, bubbles.error[i]);
        sbuf[threadIdx.x] = fmax(sbuf[threadIdx.x], maxErr);
        sbuf[threadIdx.x + 2 * BLOCK_SIZE] =
            fmax(sbuf[threadIdx.x + 2 * BLOCK_SIZE], dist);
    }

    __syncthreads();
    const int warpNum = threadIdx.x >> 5;
    const int wid = threadIdx.x & 31;
    if (threadIdx.x < 32) {
        reduce(sbuf, warpNum, &max);
        if (0 == wid) {
            maximums[blockIdx.x] = sbuf[threadIdx.x];
        }
    } else if (threadIdx.x < 64) {
        reduce(&sbuf[1 * BLOCK_SIZE], warpNum, &max);
        if (0 == wid) {
            maximums[blockIdx.x + gridDim.x] =
                sbuf[threadIdx.x + 1 * BLOCK_SIZE];
        }
    } else if (threadIdx.x < 96) {
        reduce(&sbuf[2 * BLOCK_SIZE], warpNum, &max);
        if (0 == wid) {
            maximums[blockIdx.x + 2 * gridDim.x] =
                sbuf[threadIdx.x + 2 * BLOCK_SIZE];
        }
    } else if (threadIdx.x < 128) {
        reduce(&sbuf[3 * BLOCK_SIZE], warpNum, &sum);
        if (0 == wid) {
            atomicAdd(&dTotalVolumeNew, sbuf[threadIdx.x + 3 * BLOCK_SIZE]);
        }
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

    const int warpNum = tid >> 5;
    const int wid = tid & 31;
    if (tid < 32) {
        reduce(tbds, warpNum, &sum);
        if (0 == wid) {
            atomicAdd(&dNumPairsNew, -tbds[0]);
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
    // If phi is very small, there could be zero pairs, but this loop is assumed
    // to run at least bubbles.count - dNumToBeDeleted times for adding the
    // volume
    const int bcn = bubbles.count - dNumToBeDeleted;
    const int n = dNumPairsNew > bcn ? dNumPairsNew : bcn;
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        if (i < dNumPairsNew) {
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
            // if either of the indices is a value that is beyond the new range
            // of indices, change the value to the index it was swapped to by
            // the swapDataCountPairs kernel.
            idx1 = pairs.i[i];
            idx2 = pairs.j[i];
            j = 0;
            while (j < dNumToBeDeleted) {
                // The old, swapped indices were stored after the deleted
                // indices
                int swapped = toBeDeleted[dNumToBeDeleted + j];
                if (idx1 == swapped) {
                    pairs.i[i] = toBeDeleted[j];
                } else if (idx2 == swapped) {
                    pairs.j[i] = toBeDeleted[j];
                }

                j += 1;
            }
        }

        if (i < bcn) {
            bubbles.r[i] = bubbles.r[i] * volMul;
        }
    }
}

__global__ void euler(double ts, Bubbles bubbles) {
    // Euler integration
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += blockDim.x * gridDim.x) {
        bubbles.xp[i] += bubbles.dxdtp[i] * ts;
        bubbles.yp[i] += bubbles.dydtp[i] * ts;
        if (dConstants->dimensionality == 3) {
            bubbles.zp[i] += bubbles.dzdtp[i] * ts;
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

__device__ dvec wrappedDifference(double x1, double y1, double z1, double x2,
                                  double y2, double z2) {
    dvec d1 = dvec(x1 - x2, y1 - y2, 0.0);
    if (3 == dConstants->dimensionality) {
        d1.z = z1 - z2;
    }
    dvec d2 = d1;
    dvec temp = dConstants->interval - d1.getAbsolute();
    if (!dConstants->xWall && temp.x * temp.x < d1.x * d1.x) {
        d2.x = temp.x * (d1.x < 0 ? 1.0 : -1.0);
    }
    if (!dConstants->yWall && temp.y * temp.y < d1.y * d1.y) {
        d2.y = temp.y * (d1.y < 0 ? 1.0 : -1.0);
    }
    if (3 == dConstants->dimensionality && !dConstants->zWall &&
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
