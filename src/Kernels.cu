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
__device__ double dMaxRadius;
__device__ bool dErrorEncountered;
__device__ int dNumPairs;
__device__ int dNumIncomingExternalPairs;
__device__ int dNumOutgoingExternalPairs;
__device__ int dNumPairsNew;
__device__ int dNumToBeDeleted;
}; // namespace cubble

namespace cubble {
// ==============================================
// Integration
// ==============================================
__global__ void preIntegrate(double ts, bool useGasExchange, Bubbles bubbles,
                             double *temp1) {
    // Adams-Bashforth integration
    if (blockIdx.x * blockDim.x < bubbles.count / 2) {
        __shared__ double sbuf[2 * BLOCK_SIZE];
        sbuf[threadIdx.x + 0 * BLOCK_SIZE] = 0.0;
        sbuf[threadIdx.x + 1 * BLOCK_SIZE] = 0.0;

        auto predict = [&ts](double p, double v, double vo) -> double {
            return p + 0.5 * ts * (3.0 * v - vo);
        };

        for (int i = threadIdx.x + blockIdx.x * blockDim.x;
             i < bubbles.count / 2; i += blockDim.x * gridDim.x) {
            // Every thread performs operations for two bubbles
            reinterpret_cast<double2 *>(temp1)[i] = make_double2(0.0, 0.0);
            reinterpret_cast<double2 *>(bubbles.flowVx)[i] =
                make_double2(0.0, 0.0);
            reinterpret_cast<double2 *>(bubbles.flowVy)[i] =
                make_double2(0.0, 0.0);
            reinterpret_cast<double2 *>(bubbles.flowVz)[i] =
                make_double2(0.0, 0.0);
            reinterpret_cast<double2 *>(bubbles.dxdtp)[i] =
                make_double2(0.0, 0.0);
            reinterpret_cast<double2 *>(bubbles.dydtp)[i] =
                make_double2(0.0, 0.0);

            double2 p = reinterpret_cast<double2 *>(bubbles.x)[i];
            double2 v = reinterpret_cast<double2 *>(bubbles.dxdt)[i];
            double2 vo = reinterpret_cast<double2 *>(bubbles.dxdto)[i];
            double2 result;
            result.x = predict(p.x, v.x, vo.x);
            result.y = predict(p.y, v.y, vo.y);
            reinterpret_cast<double2 *>(bubbles.xp)[i] = result;

            p = reinterpret_cast<double2 *>(bubbles.y)[i];
            v = reinterpret_cast<double2 *>(bubbles.dydt)[i];
            vo = reinterpret_cast<double2 *>(bubbles.dydto)[i];
            result.x = predict(p.x, v.x, vo.x);
            result.y = predict(p.y, v.y, vo.y);
            reinterpret_cast<double2 *>(bubbles.yp)[i] = result;

            if (3 == dConstants->dimensionality) {
                reinterpret_cast<double2 *>(bubbles.dzdtp)[i] =
                    make_double2(0.0, 0.0);
                p = reinterpret_cast<double2 *>(bubbles.z)[i];
                v = reinterpret_cast<double2 *>(bubbles.dzdt)[i];
                vo = reinterpret_cast<double2 *>(bubbles.dzdto)[i];
                result.x = predict(p.x, v.x, vo.x);
                result.y = predict(p.y, v.y, vo.y);
                reinterpret_cast<double2 *>(bubbles.zp)[i] = result;
            }

            if (useGasExchange) {
                reinterpret_cast<double2 *>(bubbles.drdtp)[i] =
                    make_double2(0.0, 0.0);
                p = reinterpret_cast<double2 *>(bubbles.r)[i];
                v = reinterpret_cast<double2 *>(bubbles.drdt)[i];
                vo = reinterpret_cast<double2 *>(bubbles.drdto)[i];
                result.x = predict(p.x, v.x, vo.x);
                result.y = predict(p.y, v.y, vo.y);
                reinterpret_cast<double2 *>(bubbles.rp)[i] = result;

                double2 areaPerRad =
                    make_double2(2.0 * CUBBLE_PI, 2.0 * CUBBLE_PI);
                if (3 == dConstants->dimensionality) {
                    areaPerRad.x *= 2.0 * result.x;
                    areaPerRad.y *= 2.0 * result.y;
                }
                sbuf[threadIdx.x + 0 * BLOCK_SIZE] +=
                    areaPerRad.x * result.x + areaPerRad.y * result.y;
                sbuf[threadIdx.x + 1 * BLOCK_SIZE] +=
                    areaPerRad.x + areaPerRad.y;
            }
        }

        // If there's an uneven number of bubbles, there's one bubble left after
        // the for loop
        if (0 == threadIdx.x + blockIdx.x && 1 == bubbles.count % 2) {
            const int i = bubbles.count - 1;
            temp1[i] = 0.0;
            bubbles.flowVx[i] = 0.0;
            bubbles.flowVy[i] = 0.0;
            bubbles.flowVz[i] = 0.0;
            bubbles.dxdtp[i] = 0.0;
            bubbles.dydtp[i] = 0.0;

            bubbles.xp[i] =
                predict(bubbles.x[i], bubbles.dxdt[i], bubbles.dxdto[i]);
            bubbles.yp[i] =
                predict(bubbles.y[i], bubbles.dydt[i], bubbles.dydto[i]);
            if (3 == dConstants->dimensionality) {
                bubbles.dzdtp[i] = 0.0;
                bubbles.zp[i] =
                    predict(bubbles.z[i], bubbles.dzdt[i], bubbles.dzdto[i]);
            }

            if (useGasExchange) {
                bubbles.drdtp[i] = 0.0;
                const double rp =
                    predict(bubbles.r[i], bubbles.drdt[i], bubbles.drdto[i]);
                bubbles.rp[i] = rp;

                double areaPerRad = 2.0 * CUBBLE_PI;
                if (3 == dConstants->dimensionality) {
                    areaPerRad *= 2.0 * rp;
                }
                sbuf[threadIdx.x + 0 * BLOCK_SIZE] += areaPerRad * rp;
                sbuf[threadIdx.x + 1 * BLOCK_SIZE] += areaPerRad;
            }
        }

        __syncthreads();
        if (useGasExchange) {
            const int warpNum = threadIdx.x >> 5;
            const int wid = threadIdx.x & 31;
            if (threadIdx.x < 32) {
                reduce(&sbuf[0 * BLOCK_SIZE], warpNum, &sum);
                if (0 == wid) {
                    atomicAdd(&dTotalArea, sbuf[threadIdx.x + 0 * BLOCK_SIZE]);
                }
            } else if (threadIdx.x < 64) {
                reduce(&sbuf[1 * BLOCK_SIZE], warpNum, &sum);
                if (0 == wid) {
                    atomicAdd(&dTotalAreaPerRadius,
                              sbuf[threadIdx.x + 1 * BLOCK_SIZE]);
                }
            }
        }
    }
}

__global__ void pairwiseInteraction(Bubbles bubbles, Pairs pairs,
                                    double *overlap, bool useGasExchange,
                                    bool useFlow) {
    // This kernel calculates both, the pairwise gas exchange and the pairwise
    // velocity.
    if (blockIdx.x * blockDim.x < dNumPairs) {
        extern __shared__ double sbuf[];
        if (useGasExchange) {
            int offset = BLOCK_SIZE * (dConstants->dimensionality + 1);
            sbuf[threadIdx.x + offset] = 0.0;
            offset += BLOCK_SIZE;
            sbuf[threadIdx.x + offset] = 0.0;
        }

        for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumPairs;
             i += gridDim.x * blockDim.x) {
            const int idx1 = pairs.i[i];
            const int idx2 = pairs.j[i];

            if (useFlow) {
                // flow vx, offset = (dim + 4 * useGas) * BLOCK_SIZE
                int offset = BLOCK_SIZE *
                             (dConstants->dimensionality + 4 * useGasExchange);
                sbuf[threadIdx.x + offset] = bubbles.dxdt[idx2];
                atomicAdd(&bubbles.flowVx[idx2], bubbles.dxdt[idx1]);

                // flow vy, offset = (dim + 4 * useGas + 1) * BLOCK_SIZE
                offset += BLOCK_SIZE;
                sbuf[threadIdx.x + offset] = bubbles.dydt[idx2];
                atomicAdd(&bubbles.flowVy[idx2], bubbles.dydt[idx1]);
                if (3 == dConstants->dimensionality) {
                    // flow vz, offset = (dim + 4 * useGas + 2) * BLOCK_SIZE
                    offset += BLOCK_SIZE;
                    sbuf[threadIdx.x + offset] = bubbles.dzdt[idx2];
                    atomicAdd(&bubbles.flowVz[idx2], bubbles.dzdt[idx1]);
                }
            }

            double r1 = bubbles.rp[idx1];
            double r2 = bubbles.rp[idx2];
            const double radii = r1 + r2;
            dvec distances = wrappedDifference(
                bubbles.xp[idx1], bubbles.yp[idx1], bubbles.zp[idx1],
                bubbles.xp[idx2], bubbles.yp[idx2], bubbles.zp[idx2]);
            const double magnitude = distances.getSquaredLength();

            if (radii * radii >= magnitude) {
                // Pair velocities
                distances = distances * dConstants->fZeroPerMuZero *
                            (rsqrt(magnitude) - 1.0 / radii);
                // dxdtp, offset = 0
                sbuf[threadIdx.x] = distances.x;
                atomicAdd(&bubbles.dxdtp[idx2], -distances.x);

                // dydtp, offset = BLOCK_SIZE
                sbuf[threadIdx.x + BLOCK_SIZE] = distances.y;
                atomicAdd(&bubbles.dydtp[idx2], -distances.y);
                if (3 == dConstants->dimensionality) {
                    // dzdtp, offset = 2 * BLOCK_SIZE
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

                    if (3 == dConstants->dimensionality) {
                        overlapArea *= CUBBLE_PI;
                    } else {
                        overlapArea = 2.0 * sqrt(overlapArea);
                    }

                    // overlap area, offset = dim * BLOCK_SIZE
                    int offset = dConstants->dimensionality * BLOCK_SIZE;
                    sbuf[threadIdx.x + offset] = overlapArea;
                    atomicAdd(&overlap[idx2], overlapArea);

                    r1 = 1.0 / r1;
                    r2 = 1.0 / r2;

                    // total overlap area, offset = (dim + 1) * BLOCK_SIZE
                    // with external bubbles only add once
                    offset += BLOCK_SIZE;
                    sbuf[threadIdx.x + offset] += 2.0 * overlapArea;

                    // total overlap area per radius,
                    // offset = (dim + 2) * BLOCK_SIZE
                    // with external bubbles only add once
                    offset += BLOCK_SIZE;
                    sbuf[threadIdx.x + offset] += overlapArea * (r1 + r2);

                    overlapArea *= (r2 - r1);

                    // drdtp, offset = (dim + 3) * BLOCK_SIZE
                    offset += BLOCK_SIZE;
                    sbuf[threadIdx.x + offset] = overlapArea;
                    atomicAdd(&bubbles.drdtp[idx2], -overlapArea);
                }
            } else {
                continue;
            }

            const unsigned int active = __activemask();
            __syncwarp(active);

            double vx = 0.0;
            double vy = 0.0;
            double vz = 0.0;
            double fvx = 0.0;
            double fvy = 0.0;
            double fvz = 0.0;
            double oa = 0.0;
            double vr = 0.0;
            const int flowOffset =
                dConstants->dimensionality + 4 * useGasExchange;
            warpReduceAtomicAddMatching(
                active, idx1, &sum<double>, sbuf, &bubbles.dxdtp[idx1], &vx, 0,
                true, &bubbles.dydtp[idx1], &vy, 1, true, &bubbles.dzdtp[idx1],
                &vz, 2, 3 == dConstants->dimensionality, &bubbles.flowVx[idx1],
                &fvx, flowOffset, useFlow, &bubbles.flowVy[idx1], &fvy,
                flowOffset + 1, useFlow, &bubbles.flowVz[idx1], &fvz,
                flowOffset + 2, (useFlow && 3 == dConstants->dimensionality),
                &overlap[idx1], &oa, dConstants->dimensionality, useGasExchange,
                &bubbles.drdtp[idx1], &vr, dConstants->dimensionality + 3,
                useGasExchange);

            __syncwarp(active);
        }

        __syncthreads();
        if (useGasExchange) {
            const int warpNum = threadIdx.x >> 5;
            const int wid = threadIdx.x & 31;
            if (threadIdx.x < 32) {
                reduce(&sbuf[(dConstants->dimensionality + 1) * BLOCK_SIZE],
                       warpNum, &sum);
                if (0 == wid) {
                    atomicAdd(
                        &dTotalOverlapArea,
                        sbuf[threadIdx.x +
                             (dConstants->dimensionality + 1) * BLOCK_SIZE]);
                }
            } else if (threadIdx.x < 64) {
                reduce(&sbuf[(dConstants->dimensionality + 2) * BLOCK_SIZE],
                       warpNum, &sum);
                if (0 == wid) {
                    atomicAdd(
                        &dTotalOverlapAreaPerRadius,
                        sbuf[threadIdx.x +
                             (dConstants->dimensionality + 2) * BLOCK_SIZE]);
                }
            }
        }
    }
}

__global__ void postIntegrate(double ts, bool useGasExchange,
                              bool incrementPath, bool useFlow, Bubbles bubbles,
                              double *blockMax, double *overlap,
                              int *toBeDeleted) {
    // This kernel applies all per bubble computations that are to be done after
    // the pairwise interactions have been calculated. These include imposed
    // flow velocity, wall velocity and liquid mediated gas exchange. After all
    // these have been computed, the prediction of each bubble is corrected
    // using Adams-Moulton integration.

    if (blockIdx.x * blockDim.x < bubbles.count) {
        __shared__ double sbuf[4 * BLOCK_SIZE];
        sbuf[threadIdx.x + 0 * BLOCK_SIZE] = 0.0;
        sbuf[threadIdx.x + 1 * BLOCK_SIZE] = 0.0;
        sbuf[threadIdx.x + 2 * BLOCK_SIZE] = 0.0;
        sbuf[threadIdx.x + 3 * BLOCK_SIZE] = 0.0;

        for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
             i += blockDim.x * gridDim.x) {
            if (useFlow) {
                addFlowVelocity(bubbles, i);
            }

            if (dConstants->xWall || dConstants->yWall || dConstants->zWall) {
                addWallVelocity(bubbles, i);
            }

            double temp = 0.0;
            double dist = 0.0;

            // X
            temp = correct(i, ts, bubbles.xp, bubbles.x, bubbles.dxdt,
                           bubbles.dxdtp, bubbles.savedX, sbuf);
            dist += temp * temp;

            // Y
            temp = correct(i, ts, bubbles.yp, bubbles.y, bubbles.dydt,
                           bubbles.dydtp, bubbles.savedY, sbuf);
            dist += temp * temp;

            // Z
            if (3 == dConstants->dimensionality) {
                temp = correct(i, ts, bubbles.zp, bubbles.z, bubbles.dzdt,
                               bubbles.dzdtp, bubbles.savedZ, sbuf);
                dist += temp * temp;
            }

            // boundary expansion = sqrt(x*x + y*y + z*z)
            dist = sqrt(dist);

            // R
            if (useGasExchange) {
                // First update the drdtp by the liquid mediated gas exchange
                double r = bubbles.rp[i];
                temp = 2.0 * CUBBLE_PI * r;
                if (3 == dConstants->dimensionality) {
                    temp *= 2.0 * r;
                }

                // vr = 1.0 / rho - 1.0 / r
                double vr = (dTotalAreaPerRadius - dTotalOverlapAreaPerRadius) /
                                (dTotalArea - dTotalOverlapArea) -
                            1.0 / r;
                // vr = vr * kappa * <A_in> / <A> * (A_free)
                vr *= dConstants->kappa * dConstants->averageSurfaceAreaIn *
                      bubbles.count / dTotalArea * (temp - overlap[i]);
                // vr = vr + pairwise gas exchange
                vr += bubbles.drdtp[i];
                // vr = K * vr / A
                vr *= dConstants->kParameter / temp;
                bubbles.drdtp[i] = vr;

                // Correct
                dist += correct(i, ts, bubbles.rp, bubbles.r, bubbles.drdt,
                                bubbles.drdtp, bubbles.savedR, sbuf);

                // Calculate volume
                r = bubbles.rp[i];
                if (r > dConstants->minRad) {
                    // Add remaining bubbles to new total volume
                    vr = r * r;
                    if (3 == dConstants->dimensionality) {
                        vr *= r;
                    }
                    sbuf[threadIdx.x + 3 * BLOCK_SIZE] += vr;
                } else {
                    toBeDeleted[atomicAdd(&dNumToBeDeleted, 1)] = i;
                }

                // maximum radius
                sbuf[threadIdx.x + 1 * BLOCK_SIZE] =
                    fmax(sbuf[threadIdx.x + 1 * BLOCK_SIZE], r);
            }

            // Store boundary expansion
            sbuf[threadIdx.x + 2 * BLOCK_SIZE] =
                fmax(sbuf[threadIdx.x + 2 * BLOCK_SIZE], dist);

            // Path
            if (incrementPath) {
                temp = bubbles.x[i] - bubbles.xp[i];
                dist = temp * temp;

                temp = bubbles.y[i] - bubbles.yp[i];
                dist += temp * temp;

                if (3 == dConstants->dimensionality) {
                    temp = bubbles.z[i] - bubbles.zp[i];
                    dist += temp * temp;
                }

                bubbles.pathNew[i] = bubbles.path[i] + sqrt(dist);
            }

            // Store the maximum error per bubble in device memory
            // The errors are reset to zero between time steps
            bubbles.error[i] = fmax(sbuf[threadIdx.x], bubbles.error[i]);
        }

        __syncthreads();
        const int warpNum = threadIdx.x >> 5;
        const int wid = threadIdx.x & 31;
        if (threadIdx.x < 32) {
            reduce(&sbuf[0 * BLOCK_SIZE], warpNum, &max);
            if (0 == wid) {
                blockMax[blockIdx.x] = sbuf[threadIdx.x];
            }
        } else if (threadIdx.x < 64) {
            reduce(&sbuf[1 * BLOCK_SIZE], warpNum, &max);
            if (0 == wid) {
                blockMax[blockIdx.x + GRID_SIZE] =
                    sbuf[threadIdx.x + 1 * BLOCK_SIZE];
            }
        } else if (threadIdx.x < 96) {
            reduce(&sbuf[2 * BLOCK_SIZE], warpNum, &max);
            if (0 == wid) {
                blockMax[blockIdx.x + 2 * GRID_SIZE] =
                    sbuf[threadIdx.x + 2 * BLOCK_SIZE];
            }
        } else if (threadIdx.x < 128) {
            reduce(&sbuf[3 * BLOCK_SIZE], warpNum, &sum);
            if (0 == wid) {
                atomicAdd(&dTotalVolumeNew, sbuf[threadIdx.x + 3 * BLOCK_SIZE]);
            }
        }
    }
}

__device__ double correct(int i, double ts, double *pp, double *p, double *v,
                          double *vp, double *old, double *maxErr) {
    const double predicted = pp[i];
    const double corrected = p[i] + 0.5 * ts * (v[i] + vp[i]);
    pp[i] = corrected;
    maxErr[threadIdx.x] = fmax(maxErr[threadIdx.x], abs(predicted - corrected));

    return corrected - old[i];
}

__device__ void addFlowVelocity(Bubbles &bubbles, int i) {
    double multiplier = bubbles.numNeighbors[i];
    multiplier = (multiplier > 0 ? 1.0 / multiplier : 0.0);
    const double xi = bubbles.xp[i];
    const double yi = bubbles.yp[i];
    double ri = bubbles.rp[i];

    int inside = (int)(xi + ri > dConstants->flowLbb.x &&
                       xi - ri < dConstants->flowTfr.x);
    inside *= (int)(yi + ri > dConstants->flowLbb.y &&
                    yi - ri < dConstants->flowTfr.y);

    if (3 == dConstants->dimensionality) {
        const double zi = bubbles.zp[i];
        inside *= (int)(zi + ri > dConstants->flowLbb.z &&
                        zi - ri < dConstants->flowTfr.z);

        bubbles.dzdtp[i] += !inside * multiplier * bubbles.flowVz[i] +
                            dConstants->flowVel.z * inside;
    }

    // Either add the average velocity of neighbors or the imposed flow,
    // if the bubble is inside the flow area
    bubbles.dxdtp[i] += !inside * multiplier * bubbles.flowVx[i] +
                        dConstants->flowVel.x * inside;
    bubbles.dydtp[i] += !inside * multiplier * bubbles.flowVy[i] +
                        dConstants->flowVel.y * inside;
}

__device__ void addWallVelocity(Bubbles &bubbles, int i) {
    const double drag = 1.0 - dConstants->wallDragStrength;
    const double rad = bubbles.rp[i];
    double xDrag = 1.0;
    double yDrag = 1.0;
    double velocity = 0.0;

    auto touchesWall = [&i, &rad, &velocity](double x, double low,
                                             double high) -> bool {
        double d1 = x - low;
        double d2 = x - high;
        d1 = d1 * d1 < d2 * d2 ? d1 : d2;
        if (rad * rad >= d1 * d1) {
            d2 = d1 < 0.0 ? -1.0 : 1.0;
            d1 *= d2;
            velocity = d2 * dConstants->fZeroPerMuZero * (1.0 - d1 / rad);
            return true;
        }

        return false;
    };

    velocity = 0.0;
    if (dConstants->xWall && touchesWall(bubbles.xp[i], dConstants->globalLbb.x,
                                         dConstants->globalTfr.x)) {
        bubbles.dxdtp[i] += velocity;
        bubbles.dydtp[i] *= drag;
        bubbles.dzdtp[i] *= drag;
        xDrag = drag;
    }

    velocity = 0.0;
    if (dConstants->yWall && touchesWall(bubbles.yp[i], dConstants->globalLbb.y,
                                         dConstants->globalTfr.y)) {
        bubbles.dxdtp[i] *= drag;
        bubbles.dydtp[i] += velocity * xDrag;
        bubbles.dzdtp[i] *= drag;
        yDrag = drag;
    }

    velocity = 0.0;
    if (dConstants->zWall && touchesWall(bubbles.zp[i], dConstants->globalLbb.z,
                                         dConstants->globalTfr.z)) {
        bubbles.dxdtp[i] *= drag;
        bubbles.dydtp[i] *= drag;
        bubbles.dzdtp[i] += velocity * xDrag * yDrag;
    }
}

// ==============================================
// Neighbor search
// ==============================================
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

__global__ void findSurfaceCells(int count, int *surfaceCells, int *cellSizes,
                                 int *surfaceCellSizes, int *bubbleCountPerArea,
                                 ivec cellDim) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < count;
         i += gridDim.x * blockDim.x) {
        const ivec ci3d = get3DIdxFrom1DIdx(i, cellDim);
        const int mx = cellDim.x - 1;
        const int my = cellDim.y - 1;
        const int mz = cellDim.z - 1;
        int idx = 0;
        const int size = cellSizes[i];

        // Add the indices of the surface cells to array
        if (3 == dConstants->dimensionality) {
            const int sideTotal =
                2 * (cellDim.x * cellDim.y + cellDim.x * cellDim.z +
                     cellDim.y * cellDim.z);
            const int edgeTotal = 4 * (cellDim.x + cellDim.y + cellDim.z);
            // Side planes
            // First x = 0 plane, then x = max plane, then the same for y
            // and z
            // x
            if (0 == ci3d.x) {
                idx = ci3d.y * cellDim.z + ci3d.z;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[0], size);
            }
            if (mx == ci3d.x) {
                idx = cellDim.y * cellDim.z + ci3d.y * cellDim.z + ci3d.z;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[1], size);
            }

            // y
            if (0 == ci3d.y) {
                idx = 2 * cellDim.y * cellDim.z + ci3d.z * cellDim.x + ci3d.x;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[2], size);
            }
            if (my == ci3d.y) {
                idx = 2 * cellDim.y * cellDim.z + cellDim.x * cellDim.z +
                      ci3d.z * cellDim.x + ci3d.x;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[3], size);
            }

            // z
            if (0 == ci3d.z) {
                idx = 2 * (cellDim.y * cellDim.z + cellDim.x * cellDim.z) +
                      ci3d.y * cellDim.x + ci3d.x;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[4], size);
            }
            if (mz == ci3d.z) {
                idx = 2 * (cellDim.y * cellDim.z + cellDim.x * cellDim.z) +
                      cellDim.x * cellDim.y + ci3d.y * cellDim.x + ci3d.x;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[5], size);
            }

            // Edges of the box
            // First use x as the rotation axis, then y, then z. In other
            // words, the first 4 edges are the one with x as the free
            // parameter, then y, then z. Sweeps start from the edge that
            // contains the (0, 0, 0) vertex.

            // x-sweep
            if (0 == ci3d.y && 0 == ci3d.z) {
                idx = sideTotal + ci3d.x;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[6], size);
            }
            if (0 == ci3d.y && mz == ci3d.z) {
                idx = sideTotal + cellDim.x + ci3d.x;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[7], size);
            }
            if (my == ci3d.y && mz == ci3d.z) {
                idx = sideTotal + 2 * cellDim.x + ci3d.x;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[8], size);
            }
            if (my == ci3d.y && 0 == ci3d.z) {
                idx = sideTotal + 3 * cellDim.x + ci3d.x;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[9], size);
            }

            // y-sweep
            if (0 == ci3d.x && 0 == ci3d.z) {
                idx = sideTotal + 4 * cellDim.x + ci3d.y;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[10], size);
            }
            if (0 == ci3d.x && mz == ci3d.z) {
                idx = sideTotal + 4 * cellDim.x + 1 * cellDim.y + ci3d.y;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[11], size);
            }
            if (mx == ci3d.x && mz == ci3d.z) {
                idx = sideTotal + 4 * cellDim.x + 2 * cellDim.y + ci3d.y;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[12], size);
            }
            if (mx == ci3d.x && 0 == ci3d.z) {
                idx = sideTotal + 4 * cellDim.x + 3 * cellDim.y + ci3d.y;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[13], size);
            }

            // z-sweep
            if (0 == ci3d.x && 0 == ci3d.y) {
                idx = sideTotal + 4 * (cellDim.x + cellDim.y) + ci3d.z;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[14], size);
            }
            if (mx == ci3d.x && 0 == ci3d.y) {
                idx = sideTotal + 4 * (cellDim.x + cellDim.y) + cellDim.z +
                      ci3d.z;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[15], size);
            }
            if (mx == ci3d.x && my == ci3d.y) {
                idx = sideTotal + 4 * (cellDim.x + cellDim.y) + 2 * cellDim.z +
                      ci3d.z;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[16], size);
            }
            if (0 == ci3d.x && my == ci3d.y) {
                idx = sideTotal + 4 * (cellDim.x + cellDim.y) + 3 * cellDim.z +
                      ci3d.z;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[17], size);
            }

            // Corners of the box
            // Start from the corners on the y = 0 plane from (0, 0, 0)
            // corner and rotating to positive direction. After the last
            // corner of the lower plane, go to the vertex above and do the
            // same for upper plane corners.
            if (0 == ci3d.x && 0 == ci3d.y && 0 == ci3d.z) {
                idx = sideTotal + edgeTotal;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[18], size);
            }
            if (0 == ci3d.x && 0 == ci3d.y && mz == ci3d.z) {
                idx = sideTotal + edgeTotal + 1;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[19], size);
            }
            if (mx == ci3d.x && 0 == ci3d.y && mz == ci3d.z) {
                idx = sideTotal + edgeTotal + 2;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[20], size);
            }
            if (mx == ci3d.x && 0 == ci3d.y && 0 == ci3d.z) {
                idx = sideTotal + edgeTotal + 3;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[21], size);
            }
            if (mx == ci3d.x && my == ci3d.y && 0 == ci3d.z) {
                idx = sideTotal + edgeTotal + 4;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[22], size);
            }
            if (0 == ci3d.x && my == ci3d.y && 0 == ci3d.z) {
                idx = sideTotal + edgeTotal + 5;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[23], size);
            }
            if (0 == ci3d.x && my == ci3d.y && mz == ci3d.z) {
                idx = sideTotal + edgeTotal + 6;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[24], size);
            }
            if (mx == ci3d.x && my == ci3d.y && mz == ci3d.z) {
                idx = sideTotal + edgeTotal + 7;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[25], size);
            }
        } else {
            if (0 == ci3d.x) {
                idx = ci3d.y;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[0], size);
                if (0 == ci3d.y) {
                    idx = 2 * (cellDim.x + cellDim.y);
                    surfaceCells[idx] = i;
                    surfaceCellSizes[idx] = size;
                    atomicAdd(&bubbleCountPerArea[26], size);
                    atomicAdd(&bubbleCountPerArea[4], size);
                } else if (my == ci3d.y) {
                    idx = 2 * (cellDim.x + cellDim.y) + 3;
                    surfaceCells[idx] = i;
                    surfaceCellSizes[idx] = size;
                    atomicAdd(&bubbleCountPerArea[26], size);
                    atomicAdd(&bubbleCountPerArea[7], size);
                }
            } else if (mx == ci3d.x) {
                idx = cellDim.y + ci3d.y;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[1], size);
                if (0 == ci3d.y) {
                    idx = 2 * (cellDim.x + cellDim.y) + 1;
                    surfaceCells[idx] = i;
                    surfaceCellSizes[idx] = size;
                    atomicAdd(&bubbleCountPerArea[26], size);
                    atomicAdd(&bubbleCountPerArea[5], size);
                } else if (my == ci3d.y) {
                    idx = 2 * (cellDim.x + cellDim.y) + 2;
                    surfaceCells[idx] = i;
                    surfaceCellSizes[idx] = size;
                    atomicAdd(&bubbleCountPerArea[26], size);
                    atomicAdd(&bubbleCountPerArea[6], size);
                }
            }

            if (0 == ci3d.y) {
                idx = 2 * cellDim.y + ci3d.x;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[2], size);
            } else if (my == ci3d.y) {
                idx = 2 * cellDim.y + cellDim.x + ci3d.x;
                surfaceCells[idx] = i;
                surfaceCellSizes[idx] = size;
                atomicAdd(&bubbleCountPerArea[26], size);
                atomicAdd(&bubbleCountPerArea[3], size);
            }
        }
    }
}

__global__ void gatherSurfaceBubbles(int count, int *surfaceCells,
                                     int *surfaceCellOffsets, int *cellSizes,
                                     int *cellOffsets, int *bubbleCounts,
                                     char **outData, Bubbles bubbles,
                                     ivec cellDim) {
    auto getNumCellsInArea = [&cellDim](int nArea) {
        int numCells = 0;
        if (3 == dConstants->dimensionality) {
            switch (nArea) {
            case 0:
            case 1:
                numCells = cellDim.y * cellDim.z;
                break;
            case 2:
            case 3:
                numCells = cellDim.x * cellDim.z;
                break;
            case 4:
            case 5:
                numCells = cellDim.x * cellDim.y;
                break;
            case 6:
            case 7:
            case 8:
            case 9:
                numCells = cellDim.x;
                break;
            case 10:
            case 11:
            case 12:
            case 13:
                numCells = cellDim.y;
                break;
            case 14:
            case 15:
            case 16:
            case 17:
                numCells = cellDim.z;
                break;
            case 18:
            case 19:
            case 20:
            case 21:
            case 22:
            case 23:
            case 24:
            case 25:
                numCells = 1;
                break;
            default:
                printf("Should never end up here at %s:%d\n", __FILE__,
                       __LINE__);
                break;
            }
        } else {
            switch (nArea) {
            case 0:
            case 1:
                numCells = cellDim.y;
                break;
            case 2:
            case 3:
                numCells = cellDim.x;
                break;
            case 4:
            case 5:
            case 6:
            case 7:
                numCells = 1;
                break;
            default:
                printf("Should never end up here at %s:%d\n", __FILE__,
                       __LINE__);
                break;
            }
        }
        return numCells;
    };

    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < count;
         i += gridDim.x * blockDim.x) {
        int indexOfFirstCellOfArea = 0;
        auto getAreaIndex = [&i, &indexOfFirstCellOfArea,
                             &getNumCellsInArea]() {
            int n = 0;
            int total = getNumCellsInArea(0);
            while (i >= total) {
                total += getNumCellsInArea(++n);
            }

            indexOfFirstCellOfArea = total - getNumCellsInArea(n);

            return n;
        };

        const int ai = getAreaIndex();
        const int nb = bubbleCounts[ai];
        char *dst = outData[ai];
        if (i == indexOfFirstCellOfArea) {
            // Add two pieces of metadata at the start
            reinterpret_cast<int *>(dst)[0] = getNumCellsInArea(ai);
            reinterpret_cast<int *>(dst)[1] = nb;
        }
        dst += 2 * sizeof(int);

        const int ci = surfaceCells[i];
        const int size = cellSizes[ci];
        const int co = cellOffsets[ci];
        // sco starts from 0 for each area
        const int sco =
            surfaceCellOffsets[i] - surfaceCellOffsets[indexOfFirstCellOfArea];

        for (int j = 0; j < size; j++) {
            const int bi = co + j;
            const int sbi = sco + j;
            int to = sbi;
            reinterpret_cast<double *>(dst)[to] = bubbles.x[bi];
            to = nb + sbi;
            reinterpret_cast<double *>(dst)[to] = bubbles.y[bi];
            to = 2 * nb + sbi;
            reinterpret_cast<double *>(dst)[to] = bubbles.z[bi];
            to = 3 * nb + sbi;
            reinterpret_cast<double *>(dst)[to] = bubbles.r[bi];
            // doubles take twice the amount of bytes cmp to int,
            // so double the base offset from 4 to 8
            to = 8 * nb + sbi;
            reinterpret_cast<int *>(dst)[to] = bi;
        }
        int to = 9 * nb + i - indexOfFirstCellOfArea;
        reinterpret_cast<int *>(dst)[to] = size;
    }
}

__global__ void scatterSurfaceBubbles(int count, char **inData,
                                      SurfaceData::Data outData) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < count;
         i += gridDim.x * blockDim.x) {
        int indexOfFirstBubbleOfArea = 0;
        int externalCellIndex = 0;
        int indexOfFirstBubbleOfCell = 0;
        auto getAreaIndex = [&i, &indexOfFirstBubbleOfArea, &inData]() {
            int n = 0;
            int total = reinterpret_cast<int *>(inData[n])[1];
            while (i >= total) {
                total += reinterpret_cast<int *>(inData[++n])[1];
            }

            indexOfFirstBubbleOfArea =
                total - reinterpret_cast<int *>(inData[n])[1];

            return n;
        };

        auto getCellIndex = [&i, &inData, &externalCellIndex,
                             &indexOfFirstBubbleOfCell](int ai, int nb) {
            int *cellSizes = reinterpret_cast<int *>(
                inData[ai] + nb * (sizeof(double) * 4 + sizeof(int)));

            // First count the number of cells and bubbles in the areas before
            // the area this bubble belongs to
            indexOfFirstBubbleOfCell = 0;
            externalCellIndex = 0;
            for (int j = 0; j < ai; j++) {
                externalCellIndex += reinterpret_cast<int *>(inData[j])[0];
                indexOfFirstBubbleOfCell +=
                    reinterpret_cast<int *>(inData[j])[1];
            }

            // Then loop over each cell that belongs to this area
            int ci = 0;
            indexOfFirstBubbleOfCell += cellSizes[ci];
            while (i >= indexOfFirstBubbleOfCell) {
                indexOfFirstBubbleOfCell += cellSizes[++ci];
            }
            indexOfFirstBubbleOfCell -= cellSizes[ci];

            return ci;
        };

        const int ai = getAreaIndex();
        const int bi = i - indexOfFirstBubbleOfArea;
        const int nb = reinterpret_cast<int *>(inData[ai])[1];
        const int ci = getCellIndex(ai, nb);
        void *src = static_cast<void *>(inData[ai] + sizeof(int) * 2);

        int from = bi;
        outData.x[i] = static_cast<double *>(src)[from];

        from = bi + nb;
        outData.y[i] = static_cast<double *>(src)[from];

        from = bi + 2 * nb;
        outData.z[i] = static_cast<double *>(src)[from];

        from = bi + 3 * nb;
        outData.r[i] = static_cast<double *>(src)[from];

        from = bi + 8 * nb;
        outData.idx[i] = static_cast<int *>(src)[from];

        // If this is the first bubble in this cell, store the size of the cell
        // in the array
        if (i == indexOfFirstBubbleOfCell) {
            from = ci + 9 * nb;
            outData.cellSizes[ci + externalCellIndex] =
                static_cast<int *>(src)[from];
        }
    }
}

__global__ void neighborSearch(int numCells, bool internalSearch,
                               int numNeighborCells, ivec cellDim, int *offsets,
                               int *sizes, int *histogram, int *pairI,
                               int *pairJ, Bubbles bubbles,
                               ExternalBubbles::Data externalBubbles,
                               SurfaceData::Data surfaceData, int *surfaceCells,
                               int *areaToProcessor) {
    // Loop over each cell pair in the simulation box
    for (int i = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
         i < numCells * numNeighborCells; i += (blockDim.x * gridDim.x) / 32) {
        // cellIdx = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, ...,
        // 99, 99, 99, 99, 99],
        // while
        // j = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, ...,
        // 0, 1, 2, 3, 4], with numCells = 100, numNeighborCells = 5

        // If this is external search, take the cell indices from the
        // surfaceCells array
        int cellIdx = i / numNeighborCells;
        if (false == internalSearch) {
            cellIdx = surfaceCells[cellIdx];
        }
        const int j = i % numNeighborCells;

        const int s1 = sizes[cellIdx];
        if (0 == s1) {
            continue;
        }

        int areaIndex = -1;
        const int neighborCellIdx = findNeighborCellIndex(
            cellIdx, cellDim, j, areaIndex, internalSearch);
        if (neighborCellIdx < 0) {
            continue;
        }

        // Choose secondary arrays based on internal/external search
        int *sizes2 = sizes;
        int *offsets2 = offsets;
        double *x2 = bubbles.x;
        double *y2 = bubbles.y;
        double *z2 = bubbles.z;
        double *r2 = bubbles.r;
        if (false == internalSearch) {
            sizes2 = surfaceData.cellSizes;
            offsets2 = surfaceData.cellOffsets;
            x2 = surfaceData.x;
            y2 = surfaceData.y;
            z2 = surfaceData.z;
            r2 = surfaceData.r;
        }
        const int s2 = sizes2[neighborCellIdx];
        if (0 == s2) {
            continue;
        }

        const int o1 = offsets[cellIdx];
        const int o2 = offsets2[neighborCellIdx];
        int numPairs = s1 * s2;
        if (neighborCellIdx == cellIdx && internalSearch) {
            // Comparing the cell to itself
            numPairs = (s1 * (s1 - 1)) / 2;
        }

        // Loop over each possible pair of bubbles in the cells
        // The whole warp has the same values for cellIdx and j
        for (int k = threadIdx.x % 32; k < numPairs; k += 32) {
            int b1 = 0;
            int b2 = 0;

            // If we're comparing a cell to itself, only compare the
            // "upper triangle" of values. Otherwise, compare all bubbles
            // of one cell to all bubbles of the other cell.
            // Insert the formula below to e.g. iPython to see what it
            // gives.
            if (neighborCellIdx == cellIdx && internalSearch) {
                b1 = s1 - 2 -
                     (int)floor(sqrt(-8.0 * k + 4 * s1 * (s1 - 1) - 7) * 0.5 -
                                0.5);
                b2 = o1 + k + b1 + 1 - s1 * (s1 - 1) / 2 +
                     (s1 - b1) * ((s1 - b1) - 1) / 2;
                b1 += o1;
            } else {
                b1 = o1 + k / s2;
                b2 = o2 + k % s2;
            }

            const double maxDistance =
                bubbles.r[b1] + r2[b2] + dConstants->skinRadius;
            if (wrappedDifference(bubbles.x[b1], bubbles.y[b1], bubbles.z[b1],
                                  x2[b2], y2[b2], z2[b2])
                    .getSquaredLength() < maxDistance * maxDistance) {
                if (internalSearch) {
                    // Set the smaller idx to b1 and larger to b2
                    int id = b1 > b2 ? b1 : b2;
                    b1 = b1 < b2 ? b1 : b2;
                    b2 = id;
                    atomicAdd(&histogram[b1], 1);
                    id = atomicAdd(&dNumPairs, 1);
                    pairI[id] = b1;
                    pairJ[id] = b2;
                } else {
                    int id = atomicAdd(&dNumIncomingExternalPairs, 1);
                    DEVICE_ASSERT(areaIndex > -1,
                                  "areaIndex is not a valid value.");
                    int pn = areaToProcessor[areaIndex];
                    atomicAdd(&histogram[pn], 1);
                    pairI[id] = b1;
                    pairJ[id] = surfaceData.idx[b2];
                    externalBubbles.procNum[id] = pn;
                    // Add the contribution of the external bubble to the number
                    // of neighbors for the local bubble
                    atomicAdd(&bubbles.numNeighbors[b1], 1);
                }
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

__global__ void sortExternalPairs(int *pairI, int *pairJ, int *procOffsets,
                                  ExternalBubbles::Data externalBubbles) {
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < dNumExternalPairs;
         i += gridDim.x * blockDim.x) {
        const int id =
            atomicSub(&procOffsets[externalBubbles.procNum[i]], 1) - 1;
        externalBubbles.internalIndex[id] = pairI[i];
        externalBubbles.externalIndex[id] = pairJ[i];
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

// ==============================================
// Bubble deletion
// ==============================================
__global__ void swapDataCountPairs(Bubbles bubbles, Pairs pairs,
                                   int *toBeDeleted) {
    // Count of pairs to be deleted
    __shared__ int tbds[BLOCK_SIZE];
    tbds[threadIdx.x] = 0;

    // The first 32 threads of the first block swap the data
    if (blockIdx.x == 0 && threadIdx.x < 32) {
        const int nNew = bubbles.count - dNumToBeDeleted;
        for (int i = threadIdx.x; i < dNumToBeDeleted; i += 32) {
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
                tbds[threadIdx.x] += 1;
            }
            j += 1;
        }
    }

    __syncthreads();

    const int warpNum = threadIdx.x >> 5;
    const int wid = threadIdx.x & 31;
    if (threadIdx.x < 32) {
        reduce(tbds, warpNum, &sum);
        if (0 == wid) {
            atomicAdd(&dNumPairsNew, -tbds[0]);
        }
    }
}

__global__ void addVolumeFixPairs(Bubbles bubbles, Pairs pairs,
                                  int *toBeDeleted) {
    double volMul = dTotalVolumeNew;
    if (3 == dConstants->dimensionality) {
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

// ==============================================
// Miscellaneous
// ==============================================
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

__global__ void euler(double ts, Bubbles bubbles) {
    // Euler integration
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += blockDim.x * gridDim.x) {
        bubbles.xp[i] += bubbles.dxdtp[i] * ts;
        bubbles.yp[i] += bubbles.dydtp[i] * ts;
        if (3 == dConstants->dimensionality) {
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
            if (3 == dConstants->dimensionality) {
                bubbles.z[i] = (bubbles.z[i] - lbb.z) / interval.z;
            }
        } else {
            bubbles.x[i] = interval.x * bubbles.x[i] + lbb.x;
            bubbles.y[i] = interval.y * bubbles.y[i] + lbb.y;
            if (3 == dConstants->dimensionality) {
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
        if (3 == dConstants->dimensionality) {
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
        if (3 == dConstants->dimensionality) {
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
        if (3 == dConstants->dimensionality) {
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

__device__ int findNeighborCellIndex(int cellIdx, ivec dim, int nn,
                                     int &areaIndex, bool internalSearch) {
    ivec idxVec = get3DIdxFrom1DIdx(cellIdx, dim);
    ivec relVec = neighborNumToRelativeCoords(nn);
    ivec oobVec = ivec(0, 0, 0);
    const bool oob = isOutOfBounds(idxVec + relVec, dim, oobVec);

    auto wrap = [&oob, &oobVec, &idxVec, &relVec]() {
        dvec glbb = dConstants->globalLbb;
        dvec gtfr = dConstants->globalTfr;
        dvec ginterval = dConstants->globalInterval;
        dvec lbb = dConstants->lbb;
        dvec tfr = dConstants->tfr;
        dvec interval = dConstants->interval;

        // Is the neighbor over the global x border?
        if ((oobVec.x < 0 && lbb.x - 1.0 < glbb.x && lbb.x + 1.0 > glbb.x) ||
            (oobVec.x > 0 && tfr.x - 1.0 < gtfr.x && tfr.x + 1.0 > gtfr.x)) {
            if (dConstants->xWall) {
                return -1;
            } else {
                // A global border and no wall. See if the local simulation
                // box spans the entire global box in this dimension. If
                // yes, wrap the coordinate.
                if (interval.x - 1.0 < ginterval.x &&
                    interval.x + 1.0 > ginterval.x) {
                    idxVec.x += relVec.x + dim.x;
                    idxVec.x %= dim.x;
                    relVec.x = 0;
                }
            }
        }

        // Is the neighbor over the global y border?
        if ((oobVec.y < 0 && lbb.y - 1.0 < glbb.y && lbb.y + 1.0 > glbb.y) ||
            (oobVec.y > 0 && tfr.y - 1.0 < gtfr.y && tfr.y + 1.0 > gtfr.y)) {
            if (dConstants->yWall) {
                return -1;
            } else {
                // A global border and no wall. See if the local simulation
                // box spans the entire global box in this dimension. If
                // yes, wrap the coordinate.
                if (interval.y - 1.0 < ginterval.y &&
                    interval.y + 1.0 > ginterval.y) {
                    idxVec.y += relVec.y + dim.y;
                    idxVec.y %= dim.y;
                    relVec.y = 0;
                }
            }
        }

        if (3 == dConstants->dimensionality) {
            // Is the neighbor over the global z border?
            if ((oobVec.z < 0 && lbb.z - 1.0 < glbb.z &&
                 lbb.z + 1.0 > glbb.z) ||
                (oobVec.z > 0 && tfr.z - 1.0 < gtfr.z &&
                 tfr.z + 1.0 > gtfr.z)) {
                if (dConstants->zWall) {
                    return -1;
                } else {
                    // A global border and no wall. See if the local simulation
                    // box spans the entire global box in this dimension. If
                    // yes, wrap the coordinate.
                    if (interval.z - 1.0 < ginterval.z &&
                        interval.z + 1.0 > ginterval.z) {
                        idxVec.z += relVec.z + dim.z;
                        idxVec.z %= dim.z;
                        relVec.z = 0;
                    }
                }
            }
        }

        return 0;
    };

    if (oob) {
        // Is there a global wall between the compared cells?
        if (wrap() == -1) {
            return -1;
        }

        // Was the cell wrapped over all the walls it went out of bounds for?
        bool allZero = relVec.x == 0;
        allZero &= relVec.y == 0;
        allZero &= relVec.z == 0;

        if (allZero) {
            // If it wrapped over all walls and we're doing internal search,
            // everything is ok, return the index of the wrapped cell
            if (internalSearch) {
                return get1DIdxFrom3DIdx(idxVec, dim);
            } else {
                // Already taken care of by the internal search
                return -1;
            }
        } else {
            if (internalSearch) {
                // If it didn't wrap around to our own box, do nothing during
                // internal search
                return -1;
            } else {
                // The neigboring cell belongs to some other processor.
                if (3 == dConstants->dimensionality) {
                    // 3D case, 26 cases in total
                    if (relVec.x < 0) {
                        // 9 different possibilities
                        if (relVec.y < 0) {
                            if (relVec.z < 0) {
                                // Corner (-1, -1, -1)
                                areaIndex = 18;
                                return 4 * (dim.x + dim.y + dim.z) +
                                       2 * (dim.z * dim.y + dim.z * dim.x +
                                            dim.x * dim.y);
                            } else if (relVec.z > 0) {
                                // Corner (-1, -1, 1)
                                areaIndex = 19;
                                return 1 + 4 * (dim.x + dim.y + dim.z) +
                                       2 * (dim.z * dim.y + dim.z * dim.x +
                                            dim.x * dim.y);
                            } else {
                                // Edge (-1, -1, z)
                                areaIndex = 14;
                                return idxVec.z + 4 * (dim.x + dim.y) +
                                       2 * (dim.z * dim.y + dim.z * dim.x +
                                            dim.x * dim.y);
                            }
                        } else if (relVec.y > 0) {
                            if (relVec.z < 0) {
                                // Corner (-1, 1, -1)
                                areaIndex = 23;
                                return 5 + 4 * (dim.x + dim.y + dim.z) +
                                       2 * (dim.z * dim.y + dim.z * dim.x +
                                            dim.x * dim.y);
                            } else if (relVec.z > 0) {
                                // Corner (-1, 1, 1)
                                areaIndex = 24;
                                return 6 + 4 * (dim.x + dim.y + dim.z) +
                                       2 * (dim.z * dim.y + dim.z * dim.x +
                                            dim.x * dim.y);
                            } else {
                                // Edge (-1, 1, z)
                                areaIndex = 17;
                                return idxVec.z + 3 * dim.z +
                                       4 * (dim.x + dim.y) +
                                       2 * (dim.z * dim.y + dim.z * dim.x +
                                            dim.x * dim.y);
                            }
                        } else if (relVec.z < 0) {
                            // Edge (-1, y, -1)
                            areaIndex = 10;
                            return idxVec.y + 4 * dim.x +
                                   2 * (dim.z * dim.y + dim.z * dim.x +
                                        dim.x * dim.y);
                        } else if (relVec.z > 0) {
                            // Edge (-1, y, 1)
                            areaIndex = 11;
                            return idxVec.y + dim.y + 4 * dim.x +
                                   2 * (dim.z * dim.y + dim.z * dim.x +
                                        dim.x * dim.y);
                        } else {
                            // Plane (-1, y, z)
                            areaIndex = 0;
                            return idxVec.y * dim.z + idxVec.z;
                        }
                    } else if (relVec.x > 0) {
                        // 9 different possibilities
                        if (relVec.y < 0) {
                            if (relVec.z < 0) {
                                // Corner (1, -1, -1)
                                areaIndex = 21;
                                return 3 + 4 * (dim.x + dim.y + dim.z) +
                                       2 * (dim.z * dim.y + dim.z * dim.x +
                                            dim.x * dim.y);
                            } else if (relVec.z > 0) {
                                // Corner (1, -1, 1)
                                areaIndex = 20;
                                return 2 + 4 * (dim.x + dim.y + dim.z) +
                                       2 * (dim.z * dim.y + dim.z * dim.x +
                                            dim.x * dim.y);
                            } else {
                                // Edge (1, -1, z)
                                areaIndex = 15;
                                return idxVec.z + dim.z + 4 * (dim.x + dim.y) +
                                       2 * (dim.z * dim.y + dim.z * dim.x +
                                            dim.x * dim.y);
                            }
                        } else if (relVec.y > 0) {
                            if (relVec.z < 0) {
                                // Corner (1, 1, -1)
                                areaIndex = 22;
                                return 4 + 4 * (dim.x + dim.y + dim.z) +
                                       2 * (dim.z * dim.y + dim.z * dim.x +
                                            dim.x * dim.y);
                            } else if (relVec.z > 0) {
                                // Corner (1, 1, 1)
                                areaIndex = 25;
                                return 7 + 4 * (dim.x + dim.y + dim.z) +
                                       2 * (dim.z * dim.y + dim.z * dim.x +
                                            dim.x * dim.y);
                            } else {
                                // Edge (1, 1, z)
                                areaIndex = 16;
                                return idxVec.z + 2 * dim.z +
                                       4 * (dim.x + dim.y) +
                                       2 * (dim.z * dim.y + dim.z * dim.x +
                                            dim.x * dim.y);
                            }
                        } else if (relVec.z < 0) {
                            // Edge (1, y, -1)
                            areaIndex = 13;
                            return idxVec.y + 3 * dim.y + 4 * dim.x +
                                   2 * (dim.z * dim.y + dim.z * dim.x +
                                        dim.x * dim.y);
                        } else if (relVec.z > 0) {
                            // Edge (1, y, 1)
                            areaIndex = 12;
                            return idxVec.y + 2 * dim.y + 4 * dim.x +
                                   2 * (dim.z * dim.y + dim.z * dim.x +
                                        dim.x * dim.y);
                        } else {
                            // Plane (1, y, z)
                            areaIndex = 1;
                            return idxVec.y * dim.z + idxVec.z + dim.z * dim.y;
                        }
                    } else if (relVec.y < 0) {
                        // 3 possibilities
                        if (relVec.z < 0) {
                            // Edge (x, -1, -1)
                            areaIndex = 6;
                            return idxVec.x +
                                   2 * (dim.z * dim.y + dim.z * dim.x +
                                        dim.x * dim.y);
                        } else if (relVec.z > 0) {
                            // Edge (x, -1, 1)
                            areaIndex = 9;
                            return idxVec.x + 3 * dim.x +
                                   2 * (dim.z * dim.y + dim.z * dim.x +
                                        dim.x * dim.y);
                        } else {
                            // Plane (x, -1, z)
                            areaIndex = 2;
                            return idxVec.z * dim.x + idxVec.x +
                                   2 * dim.z * dim.y;
                        }
                    } else if (relVec.y > 0) {
                        // 3 possibilities
                        if (relVec.z < 0) {
                            // Edge (x, 1, -1)
                            areaIndex = 7;
                            return idxVec.x + dim.x +
                                   2 * (dim.z * dim.y + dim.z * dim.x +
                                        dim.x * dim.y);
                        } else if (relVec.z > 0) {
                            // Edge (x, 1, 1)
                            areaIndex = 8;
                            return idxVec.x + 2 * dim.x +
                                   2 * (dim.z * dim.y + dim.z * dim.x +
                                        dim.x * dim.y);
                        } else {
                            // Plane (x, 1, z)
                            areaIndex = 3;
                            return idxVec.z * dim.x + idxVec.x +
                                   2 * dim.z * dim.y + dim.z * dim.x;
                        }
                    } else if (relVec.z < 0) {
                        // Plane (x, y, -1)
                        areaIndex = 4;
                        return idxVec.y * dim.x + idxVec.x +
                               2 * (dim.z * dim.y + dim.z * dim.x);
                    } else if (relVec.z > 0) {
                        // Plane (x, y, 1)
                        areaIndex = 5;
                        return idxVec.y * dim.x + idxVec.x + dim.x * dim.y +
                               2 * (dim.z * dim.y + dim.z * dim.x);
                    } else {
                        printf("Should never end up here!\n");
                    }
            } else {
                // 2D case, 8 cases in total
                if (relVec.x < 0) {
                    // 3 cases
                    if (relVec.y < 0) {
                        // Corner (-1, -1)
                        areaIndex = 4;
                        return 2 * (dim.x + dim.y);
                    } else if (relVec.y > 0) {
                        // Corner (-1, 1)
                        areaIndex = 7;
                        return 3 + 2 * (dim.x + dim.y);
                    } else {
                        // Edge (-1, y)
                        areaIndex = 0;
                        return idxVec.x;
                    }
                } else if (relVec.x > 0) {
                    // 3 cases
                    if (relVec.y < 0) {
                        // Corner (1, -1)
                        areaIndex = 5;
                        return 1 + 2 * (dim.x + dim.y);
                    } else if (relVec.y > 0) {
                        // Corner (1, 1)
                        areaIndex = 6;
                        return 2 + 2 * (dim.x + dim.y);
                    } else {
                        // Edge (1, y)
                        areaIndex = 1;
                        return idxVec.x + dim.x;
                    }
                } else if (relVec.y < 0) {
                    // Edge (x, -1)
                    areaIndex = 2;
                    return idxVec.y + 2 * dim.x;
                } else if (relVec.y > 0) {
                    // Edge (x, 1)
                    areaIndex = 3;
                    return idxVec.y + dim.y + 2 * dim.x;
                } else {
                    printf("Should never end up here!\n");
                }
            }
            }
        }
    } else {
        // The cell was inside our local box
        if (internalSearch) {
            return get1DIdxFrom3DIdx(idxVec + relVec, dim);
        } else {
            // Already taken care of by the internal search
            return -1;
        }
    }

    printf("Should never end up here!\n");
    return -1;
}

__device__ bool isOutOfBounds(ivec a, ivec dim, ivec &oobVec) {
    oobVec = ivec(0, 0, 0);
    bool oob = false;

    // x
    if (a.x < 0) {
        oobVec.x = -1;
        oob = true;
    } else if (a.x > dim.x) {
        oobVec.x = 1;
        oob = true;
    }

    // y
    if (a.y < 0) {
        oobVec.y = -1;
        oob = true;
    } else if (a.y > dim.y) {
        oobVec.y = 1;
        oob = true;
    }

    // z
    if (a.z < 0) {
        oobVec.z = -1;
        oob = true;
    } else if (a.z > dim.z) {
        oobVec.z = 1;
        oob = true;
    }

    return oob;
}

__device__ ivec neighborNumToRelativeCoords(int nn) {
    ivec idxVec = ivec(0, 0, 0);
    if (3 == dConstants->dimensionality) {
        switch (nn) {
        case 0:
            idxVec = ivec(0, 0, 0);
            break;
        case 1:
            idxVec = ivec(-1, 0, 0);
            break;
        case 2:
            idxVec = ivec(-1, 0, 1);
            break;
        case 3:
            idxVec = ivec(0, 0, 1);
            break;
        case 4:
            idxVec = ivec(1, 0, 1);
            break;
        case 5:
            idxVec = ivec(1, -1, 1);
            break;
        case 6:
            idxVec = ivec(0, -1, 1);
            break;
        case 7:
            idxVec = ivec(-1, -1, 1);
            break;
        case 8:
            idxVec = ivec(-1, -1, 0);
            break;
        case 9:
            idxVec = ivec(0, -1, 0);
            break;
        case 10:
            idxVec = ivec(1, -1, 0);
            break;
        case 11:
            idxVec = ivec(1, -1, -1);
            break;
        case 12:
            idxVec = ivec(0, -1, -1);
            break;
        case 13:
            idxVec = ivec(-1, -1, -1);
            break;
        case 14:
            idxVec = ivec(-1, 0, -1);
            break;
        case 15:
            idxVec = ivec(0, 0, -1);
            break;
        case 16:
            idxVec = ivec(1, 0, -1);
            break;
        case 17:
            idxVec = ivec(1, 0, 0);
            break;
        case 18:
            idxVec = ivec(1, 1, 0);
            break;
        case 19:
            idxVec = ivec(1, 1, -1);
            break;
        case 20:
            idxVec = ivec(0, 1, -1);
            break;
        case 21:
            idxVec = ivec(-1, 1, -1);
            break;
        case 22:
            idxVec = ivec(-1, 1, 0);
            break;
        case 23:
            idxVec = ivec(-1, 1, 1);
            break;
        case 24:
            idxVec = ivec(0, 1, 1);
            break;
        case 25:
            idxVec = ivec(1, 1, 1);
            break;
        case 26:
            idxVec = ivec(0, 1, 0);
            break;
        default:
            printf("Should never end up here!\n");
            break;
        }
    } else {
        switch (nn) {
        case 0:
            idxVec = ivec(0, 0, 0);
            break;
        case 1:
            idxVec = ivec(-1, 0, 0);
            break;
        case 2:
            idxVec = ivec(-1, -1, 0);
            break;
        case 3:
            idxVec = ivec(0, -1, 0);
            break;
        case 4:
            idxVec = ivec(1, -1, 0);
            break;
        case 5:
            idxVec = ivec(1, 0, 0);
            break;
        case 6:
            idxVec = ivec(1, 1, 0);
            break;
        case 7:
            idxVec = ivec(0, 1, 0);
            break;
        case 8:
            idxVec = ivec(-1, 1, 0);
            break;
        default:
            printf("Should never end up here!\n");
            break;
        }
    }

    return idxVec;
}

__device__ int getCellIdxFromPos(double x, double y, double z, ivec cellDim) {
    const dvec lbb = dConstants->lbb;
    const dvec interval = dConstants->interval;
    const int xid = floor(cellDim.x * (x - lbb.x) / interval.x);
    const int yid = floor(cellDim.y * (y - lbb.y) / interval.y);
    int zid = 0;
    if (3 == dConstants->dimensionality) {
        zid = floor(cellDim.z * (z - lbb.z) / interval.z);
    }

    return get1DIdxFrom3DIdx(ivec(xid, yid, zid), cellDim);
}

__device__ int get1DIdxFrom3DIdx(ivec idxVec, ivec cellDim) {
    return idxVec.z * cellDim.x * cellDim.y + idxVec.y * cellDim.x + idxVec.x;
}

__device__ ivec get3DIdxFrom1DIdx(int idx, ivec cellDim) {
    ivec idxVec(0, 0, 0);
    idxVec.x = idx % cellDim.x;
    idxVec.y = (idx / cellDim.x) % cellDim.y;
    if (3 == dConstants->dimensionality) {
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
    dMaxRadius = 0.0;
    dNumToBeDeleted = 0;
    dNumPairsNew = dNumPairs;
    dErrorEncountered = false;
}
} // namespace cubble
