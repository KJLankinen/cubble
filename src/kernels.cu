/*
    Cubble
    Copyright (C) 2024  Juhana Lankinen

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

#include "device_globals.cuh"
#include "kernels.cuh"
#include "macros.h"

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

        for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
             i < bubbles.count / 2; i += blockDim.x * gridDim.x) {
            // Every thread performs operations for two bubbles
            reinterpret_cast<double2 *>(temp1)[i] = make_double2(0.0, 0.0);
            reinterpret_cast<double2 *>(bubbles.flow_vx)[i] =
                make_double2(0.0, 0.0);
            reinterpret_cast<double2 *>(bubbles.flow_vy)[i] =
                make_double2(0.0, 0.0);
            reinterpret_cast<double2 *>(bubbles.flow_vz)[i] =
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

            if (3 == d_constants->dimensionality) {
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

                double2 area_per_rad =
                    make_double2(2.0 * CUBBLE_PI, 2.0 * CUBBLE_PI);
                if (3 == d_constants->dimensionality) {
                    area_per_rad.x *= 2.0 * result.x;
                    area_per_rad.y *= 2.0 * result.y;
                }
                sbuf[threadIdx.x + 0 * BLOCK_SIZE] +=
                    area_per_rad.x * result.x + area_per_rad.y * result.y;
                sbuf[threadIdx.x + 1 * BLOCK_SIZE] +=
                    area_per_rad.x + area_per_rad.y;
            }
        }

        // If there's an uneven number of bubbles, there's one bubble left after
        // the for loop
        if (0 == threadIdx.x + blockIdx.x && 1 == bubbles.count % 2) {
            const int32_t i = bubbles.count - 1;
            temp1[i] = 0.0;
            bubbles.flow_vx[i] = 0.0;
            bubbles.flow_vy[i] = 0.0;
            bubbles.flow_vz[i] = 0.0;
            bubbles.dxdtp[i] = 0.0;
            bubbles.dydtp[i] = 0.0;

            bubbles.xp[i] =
                predict(bubbles.x[i], bubbles.dxdt[i], bubbles.dxdto[i]);
            bubbles.yp[i] =
                predict(bubbles.y[i], bubbles.dydt[i], bubbles.dydto[i]);
            if (3 == d_constants->dimensionality) {
                bubbles.dzdtp[i] = 0.0;
                bubbles.zp[i] =
                    predict(bubbles.z[i], bubbles.dzdt[i], bubbles.dzdto[i]);
            }

            if (useGasExchange) {
                bubbles.drdtp[i] = 0.0;
                const double rp =
                    predict(bubbles.r[i], bubbles.drdt[i], bubbles.drdto[i]);
                bubbles.rp[i] = rp;

                double area_per_rad = 2.0 * CUBBLE_PI;
                if (3 == d_constants->dimensionality) {
                    area_per_rad *= 2.0 * rp;
                }
                sbuf[threadIdx.x + 0 * BLOCK_SIZE] += area_per_rad * rp;
                sbuf[threadIdx.x + 1 * BLOCK_SIZE] += area_per_rad;
            }
        }

        __syncthreads();
        if (useGasExchange) {
            const int32_t warp_num = threadIdx.x >> 5;
            const int32_t wid = threadIdx.x & 31;
            if (threadIdx.x < 32) {
                reduce(&sbuf[0 * BLOCK_SIZE], warp_num, &sum);
                if (0 == wid) {
                    atomicAdd(&d_total_area,
                              sbuf[threadIdx.x + 0 * BLOCK_SIZE]);
                }
            } else if (threadIdx.x < 64) {
                reduce(&sbuf[1 * BLOCK_SIZE], warp_num, &sum);
                if (0 == wid) {
                    atomicAdd(&d_total_area_per_radius,
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
    if (blockIdx.x * blockDim.x < d_num_pairs) {
        extern __shared__ double sbuf[];
        if (useGasExchange) {
            int32_t offset = BLOCK_SIZE * (d_constants->dimensionality + 1);
            sbuf[threadIdx.x + offset] = 0.0;
            offset += BLOCK_SIZE;
            sbuf[threadIdx.x + offset] = 0.0;
        }

        for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < d_num_pairs;
             i += gridDim.x * blockDim.x) {
            const int32_t idx1 = pairs.i[i];
            const int32_t idx2 = pairs.j[i];

            if (useFlow) {
                // flow vx, offset = (dim + 4 * useGas) * BLOCK_SIZE
                int32_t offset = BLOCK_SIZE * (d_constants->dimensionality +
                                               4 * useGasExchange);
                sbuf[threadIdx.x + offset] = bubbles.dxdt[idx2];
                atomicAdd(&bubbles.flow_vx[idx2], bubbles.dxdt[idx1]);

                // flow vy, offset = (dim + 4 * useGas + 1) * BLOCK_SIZE
                offset += BLOCK_SIZE;
                sbuf[threadIdx.x + offset] = bubbles.dydt[idx2];
                atomicAdd(&bubbles.flow_vy[idx2], bubbles.dydt[idx1]);
                if (3 == d_constants->dimensionality) {
                    // flow vz, offset = (dim + 4 * useGas + 2) * BLOCK_SIZE
                    offset += BLOCK_SIZE;
                    sbuf[threadIdx.x + offset] = bubbles.dzdt[idx2];
                    atomicAdd(&bubbles.flow_vz[idx2], bubbles.dzdt[idx1]);
                }
            }

            double r1 = bubbles.rp[idx1];
            double r2 = bubbles.rp[idx2];
            const double radii = r1 + r2;
            dvec distances = wrappedDifference(
                bubbles.xp[idx1], bubbles.yp[idx1], bubbles.zp[idx1],
                bubbles.xp[idx2], bubbles.yp[idx2], bubbles.zp[idx2]);
            const double magnitude = lengthSq(distances);

            if (radii * radii >= magnitude) {
                // Pair velocities
                distances = distances * d_constants->f_zero_per_mu_zero *
                            (rsqrt(magnitude) - 1.0 / radii);
                // dxdtp, offset = 0
                sbuf[threadIdx.x] = distances.x;
                atomicAdd(&bubbles.dxdtp[idx2], -distances.x);

                // dydtp, offset = BLOCK_SIZE
                sbuf[threadIdx.x + BLOCK_SIZE] = distances.y;
                atomicAdd(&bubbles.dydtp[idx2], -distances.y);
                if (3 == d_constants->dimensionality) {
                    // dzdtp, offset = 2 * BLOCK_SIZE
                    sbuf[threadIdx.x + 2 * BLOCK_SIZE] = distances.z;
                    atomicAdd(&bubbles.dzdtp[idx2], -distances.z);
                }

                // Pairwise gas exchange
                if (useGasExchange) {
                    const double r1sq = r1 * r1;
                    const double r2sq = r2 * r2;
                    double overlap_area = 0;
                    if (magnitude < r1sq || magnitude < r2sq) {
                        overlap_area = r1sq < r2sq ? r1sq : r2sq;
                    } else {
                        overlap_area = 0.5 * (r2sq - r1sq + magnitude);
                        overlap_area *= overlap_area;
                        overlap_area /= magnitude;
                        overlap_area = r2sq - overlap_area;
                        overlap_area =
                            overlap_area < 0 ? -overlap_area : overlap_area;
                    }

                    if (3 == d_constants->dimensionality) {
                        overlap_area *= CUBBLE_PI;
                    } else {
                        overlap_area = 2.0 * sqrt(overlap_area);
                    }

                    // overlap area, offset = dim * BLOCK_SIZE
                    int32_t offset = d_constants->dimensionality * BLOCK_SIZE;
                    sbuf[threadIdx.x + offset] = overlap_area;
                    atomicAdd(&overlap[idx2], overlap_area);

                    r1 = 1.0 / r1;
                    r2 = 1.0 / r2;

                    // total overlap area, offset = (dim + 1) * BLOCK_SIZE
                    offset += BLOCK_SIZE;
                    sbuf[threadIdx.x + offset] += 2.0 * overlap_area;

                    // total overlap area per radius,
                    // offset = (dim + 2) * BLOCK_SIZE
                    offset += BLOCK_SIZE;
                    sbuf[threadIdx.x + offset] += overlap_area * (r1 + r2);

                    overlap_area *= (r2 - r1);

                    // drdtp, offset = (dim + 3) * BLOCK_SIZE
                    offset += BLOCK_SIZE;
                    sbuf[threadIdx.x + offset] = overlap_area;
                    atomicAdd(&bubbles.drdtp[idx2], -overlap_area);
                }
            } else {
                continue;
            }

            const uint32_t active = __activemask();
            __syncwarp(active);

            double vx = 0.0;
            double vy = 0.0;
            double vz = 0.0;
            double fvx = 0.0;
            double fvy = 0.0;
            double fvz = 0.0;
            double oa = 0.0;
            double vr = 0.0;
            const int32_t flow_offset =
                d_constants->dimensionality + 4 * useGasExchange;
            warpReduceAtomicAddMatching(
                active, idx1, &sum<double>, sbuf, &bubbles.dxdtp[idx1], &vx, 0,
                true, &bubbles.dydtp[idx1], &vy, 1, true, &bubbles.dzdtp[idx1],
                &vz, 2, 3 == d_constants->dimensionality,
                &bubbles.flow_vx[idx1], &fvx, flow_offset, useFlow,
                &bubbles.flow_vy[idx1], &fvy, flow_offset + 1, useFlow,
                &bubbles.flow_vz[idx1], &fvz, flow_offset + 2,
                (useFlow && 3 == d_constants->dimensionality), &overlap[idx1],
                &oa, d_constants->dimensionality, useGasExchange,
                &bubbles.drdtp[idx1], &vr, d_constants->dimensionality + 3,
                useGasExchange);

            __syncwarp(active);
        }

        __syncthreads();
        if (useGasExchange) {
            const int32_t warp_num = threadIdx.x >> 5;
            const int32_t wid = threadIdx.x & 31;
            if (threadIdx.x < 32) {
                reduce(&sbuf[(d_constants->dimensionality + 1) * BLOCK_SIZE],
                       warp_num, &sum);
                if (0 == wid) {
                    atomicAdd(
                        &d_total_overlap_area,
                        sbuf[threadIdx.x +
                             (d_constants->dimensionality + 1) * BLOCK_SIZE]);
                }
            } else if (threadIdx.x < 64) {
                reduce(&sbuf[(d_constants->dimensionality + 2) * BLOCK_SIZE],
                       warp_num, &sum);
                if (0 == wid) {
                    atomicAdd(
                        &d_total_overlap_area_per_radius,
                        sbuf[threadIdx.x +
                             (d_constants->dimensionality + 2) * BLOCK_SIZE]);
                }
            }
        }
    }
}

__global__ void postIntegrate(double ts, bool useGasExchange,
                              bool incrementPath, bool useFlow, Bubbles bubbles,
                              double *blockMax, double *overlap,
                              int32_t *toBeDeleted) {
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

        for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x;
             i < bubbles.count; i += blockDim.x * gridDim.x) {
            if (useFlow) {
                addFlowVelocity(bubbles, i);
            }

            if (d_constants->x_wall || d_constants->y_wall ||
                d_constants->z_wall) {
                addWallVelocity(bubbles, i);
            }

            double temp = 0.0;
            double dist = 0.0;

            // X
            temp = correct(i, ts, bubbles.xp, bubbles.x, bubbles.dxdt,
                           bubbles.dxdtp, bubbles.saved_x, sbuf);
            dist += temp * temp;

            // Y
            temp = correct(i, ts, bubbles.yp, bubbles.y, bubbles.dydt,
                           bubbles.dydtp, bubbles.saved_y, sbuf);
            dist += temp * temp;

            // Z
            if (3 == d_constants->dimensionality) {
                temp = correct(i, ts, bubbles.zp, bubbles.z, bubbles.dzdt,
                               bubbles.dzdtp, bubbles.saved_z, sbuf);
                dist += temp * temp;
            }

            // boundary expansion = sqrt(x*x + y*y + z*z)
            dist = sqrt(dist);

            // R
            if (useGasExchange) {
                // First update the drdtp by the liquid mediated gas exchange
                double r = bubbles.rp[i];
                temp = 2.0 * CUBBLE_PI * r;
                if (d_constants->dimensionality == 3) {
                    temp *= 2.0 * r;
                }

                // vr = 1.0 / rho - 1.0 / r
                double vr = (d_total_area_per_radius -
                             d_total_overlap_area_per_radius) /
                                (d_total_area - d_total_overlap_area) -
                            1.0 / r;
                // vr = vr * kappa * <A_in> / <A> * (A_free)
                vr *= d_constants->kappa *
                      d_constants->average_surface_area_in * bubbles.count /
                      d_total_area * (temp - overlap[i]);
                // vr = vr + pairwise gas exchange
                vr += bubbles.drdtp[i];
                // vr = K * vr / A
                vr *= d_constants->k_parameter / temp;
                bubbles.drdtp[i] = vr;

                // Correct
                dist += correct(i, ts, bubbles.rp, bubbles.r, bubbles.drdt,
                                bubbles.drdtp, bubbles.saved_r, sbuf);

                // Calculate volume
                r = bubbles.rp[i];
                if (r > d_constants->min_rad) {
                    // Add remaining bubbles to new total volume
                    vr = r * r;
                    if (3 == d_constants->dimensionality) {
                        vr *= r;
                    }
                    sbuf[threadIdx.x + 3 * BLOCK_SIZE] += vr;
                } else {
                    toBeDeleted[atomicAdd(&d_num_to_be_deleted, 1)] = i;
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

                if (d_constants->dimensionality == 3) {
                    temp = bubbles.z[i] - bubbles.zp[i];
                    dist += temp * temp;
                }

                bubbles.path_new[i] = bubbles.path[i] + sqrt(dist);
            }

            // Store the maximum error per bubble in device memory
            // The errors are reset to zero between time steps
            bubbles.error[i] = fmax(sbuf[threadIdx.x], bubbles.error[i]);
        }

        __syncthreads();
        const int32_t warp_num = threadIdx.x >> 5;
        const int32_t wid = threadIdx.x & 31;
        if (threadIdx.x < 32) {
            reduce(&sbuf[0 * BLOCK_SIZE], warp_num, &max);
            if (0 == wid) {
                blockMax[blockIdx.x] = sbuf[threadIdx.x];
            }
        } else if (threadIdx.x < 64) {
            reduce(&sbuf[1 * BLOCK_SIZE], warp_num, &max);
            if (0 == wid) {
                blockMax[blockIdx.x + GRID_SIZE] =
                    sbuf[threadIdx.x + 1 * BLOCK_SIZE];
            }
        } else if (threadIdx.x < 96) {
            reduce(&sbuf[2 * BLOCK_SIZE], warp_num, &max);
            if (0 == wid) {
                blockMax[blockIdx.x + 2 * GRID_SIZE] =
                    sbuf[threadIdx.x + 2 * BLOCK_SIZE];
            }
        } else if (threadIdx.x < 128) {
            reduce(&sbuf[3 * BLOCK_SIZE], warp_num, &sum);
            if (0 == wid) {
                atomicAdd(&d_total_volume_new,
                          sbuf[threadIdx.x + 3 * BLOCK_SIZE]);
            }
        }
    }
}

// ==============================================
// Neighbor search
// ==============================================
__global__ void cellByPosition(int32_t *cellIndices, int32_t *cellSizes,
                               ivec cellDim, Bubbles bubbles) {
    for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += gridDim.x * blockDim.x) {
        const int32_t ci = getCellIdxFromPos(bubbles.x[i], bubbles.y[i],
                                             bubbles.z[i], cellDim);
        cellIndices[i] = ci;
        atomicAdd(&cellSizes[ci], 1);
    }
}

__global__ void indexByCell(int32_t *cellIndices, int32_t *cellOffsets,
                            int32_t *bubbleIndices, int32_t count) {
    for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < count;
         i += gridDim.x * blockDim.x) {
        bubbleIndices[atomicSub(&cellOffsets[cellIndices[i]], 1) - 1] = i;
    }
}

__global__ void neighborSearch(int32_t numCells, int32_t numNeighborCells,
                               ivec cellDim, int32_t *offsets, int32_t *sizes,
                               int32_t *histogram, int32_t *pairI,
                               int32_t *pairJ, Bubbles bubbles) {
    DEVICE_ASSERT(blockDim.x >= 32, "Use at least 32 threads.");
    // Loop over each cell in the simulation box
    for (int32_t i = blockIdx.x; i < numCells; i += gridDim.x) {
        const int32_t s1 = sizes[i];
        if (0 == s1) {
            continue;
        }
        // Loop over each neighbor-cell-to-consider
        for (int32_t j = threadIdx.x / 32; j < numNeighborCells;
             j += blockDim.x / 32) {
            const int32_t ci = getNeighborCellIndex(i, cellDim, j);
            if (ci < 0) {
                continue;
            }
            const int32_t s2 = sizes[ci];
            if (0 == s2) {
                continue;
            }

            const int32_t o1 = offsets[i];
            const int32_t o2 = offsets[ci];
            int32_t num_pairs = s1 * s2;
            if (ci == i) {
                // Comparing the cell to itself
                num_pairs = (s1 * (s1 - 1)) / 2;
            }

            // Loop over each possible pair of bubbles in the cells
            for (int32_t k = threadIdx.x % 32; k < num_pairs; k += 32) {
                int32_t b1 = 0;
                int32_t b2 = 0;

                // If we're comparing a cell to itself, only compare the
                // "upper triangle" of values. Otherwise, compare all bubbles
                // of one cell to all bubbles of the other cell.
                // Insert the formula below to e.g. iPython to see what it
                // gives.
                if (ci == i) {
                    b1 =
                        s1 - 2 -
                        (int32_t)::floor(
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

                comparePair(b1, b2, histogram, pairI, pairJ, bubbles);
                // Eh? this function doesn't take in pairs...
                // DEVICE_ASSERT(pairs.stride > d_num_pairs,
                //              "Too many neighbor indices!");
            }
        }
    }
}

__global__ void sortPairs(Bubbles bubbles, Pairs pairs, int32_t *pairI,
                          int32_t *pairJ) {
    for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < d_num_pairs;
         i += gridDim.x * blockDim.x) {
        const int32_t id = atomicSub(&bubbles.num_neighbors[pairI[i]], 1) - 1;
        pairs.i[id] = pairI[i];
        pairs.j[id] = pairJ[i];
    }
}

__global__ void countNumNeighbors(Bubbles bubbles, Pairs pairs) {
    for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < d_num_pairs;
         i += gridDim.x * blockDim.x) {
        atomicAdd(&bubbles.num_neighbors[pairs.i[i]], 1);
        atomicAdd(&bubbles.num_neighbors[pairs.j[i]], 1);
    }
}

__global__ void reorganizeByIndex(Bubbles bubbles, const int32_t *newIndex) {
    for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += gridDim.x * blockDim.x) {
        int32_t j = newIndex[i];
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
        int32_t k = newIndex[j];
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

// ==============================================
// Bubble deletion
// ==============================================
__global__ void swapDataCountPairs(Bubbles bubbles, Pairs pairs,
                                   int32_t *toBeDeleted) {
    // Count of pairs to be deleted
    __shared__ int32_t tbds[BLOCK_SIZE];
    tbds[threadIdx.x] = 0;

    // The first 32 threads of the first block swap the data
    if (blockIdx.x == 0 && threadIdx.x < 32) {
        const int32_t n_new = bubbles.count - d_num_to_be_deleted;
        for (int32_t i = threadIdx.x; i < d_num_to_be_deleted; i += 32) {
            // If the to-be-deleted index is inside the remaining indices,
            // it will be swapped with one from the back that won't be
            // removed but which is outside the new range (i.e. would be
            // erroneously removed).
            const int32_t idx1 = toBeDeleted[i];
            if (idx1 < n_new) {
                // Count how many values before this ith value are swapped
                // from the back. In other words, count how many good values
                // to skip, before we choose which one to swap with idx1.
                int32_t from_back = i;
                int32_t j = 0;
                while (j < i) {
                    if (toBeDeleted[j] >= n_new) {
                        from_back -= 1;
                    }
                    j += 1;
                }

                // Find the index idx2 of the jth remaining (i.e. "good")
                // value from the back, which still lies outside the new
                // range.
                j = 0;
                int32_t idx2 = bubbles.count - 1;
                while (idx2 >= n_new &&
                       (bubbles.r[idx2] < d_constants->min_rad ||
                        j != from_back)) {
                    if (bubbles.r[idx2] >= d_constants->min_rad) {
                        j += 1;
                    }
                    idx2 -= 1;
                }

                // Append the old indices for later use
                toBeDeleted[i + d_num_to_be_deleted] = idx2;

                // Swap all the arrays
                swapValues(idx2, idx1, bubbles.x, bubbles.y, bubbles.z,
                           bubbles.r, bubbles.dxdt, bubbles.dydt, bubbles.dzdt,
                           bubbles.drdt, bubbles.dxdto, bubbles.dydto,
                           bubbles.dzdto, bubbles.drdto, bubbles.saved_x,
                           bubbles.saved_y, bubbles.saved_z, bubbles.saved_r,
                           bubbles.path, bubbles.error, bubbles.wrap_count_x,
                           bubbles.wrap_count_y, bubbles.wrap_count_z,
                           bubbles.index, bubbles.num_neighbors);
            } else {
                toBeDeleted[i + d_num_to_be_deleted] = idx1;
            }
        }
    }

    // All threads check how many pairs are to be deleted
    for (int32_t i = threadIdx.x + blockDim.x * blockIdx.x; i < d_num_pairs;
         i += blockDim.x * gridDim.x) {
        int32_t j = 0;
        while (j < d_num_to_be_deleted) {
            const int32_t tbd = toBeDeleted[j];
            if (pairs.i[i] == tbd || pairs.j[i] == tbd) {
                tbds[threadIdx.x] += 1;
            }
            j += 1;
        }
    }

    __syncthreads();

    const int32_t warp_num = threadIdx.x >> 5;
    const int32_t wid = threadIdx.x & 31;
    if (threadIdx.x < 32) {
        reduce(tbds, warp_num, &sum);
        if (0 == wid) {
            atomicAdd(&d_num_pairs_new, -tbds[0]);
        }
    }
}

__global__ void addVolumeFixPairs(Bubbles bubbles, Pairs pairs,
                                  int32_t *toBeDeleted) {
    double vol_mul = d_total_volume_new;
    if (d_constants->dimensionality == 3) {
        vol_mul = rcbrt(vol_mul);
    } else {
        vol_mul = rsqrt(vol_mul);
    }
    vol_mul *= d_constants->bubble_volume_multiplier;
    // If phi is very small, there could be zero pairs, but this loop is assumed
    // to run at least bubbles.count - d_num_to_be_deleted times for adding the
    // volume
    const int32_t bcn = bubbles.count - d_num_to_be_deleted;
    const int32_t n = d_num_pairs_new > bcn ? d_num_pairs_new : bcn;
    for (int32_t i = threadIdx.x + blockDim.x * blockIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        if (i < d_num_pairs_new) {
            // Check if either of the indices of this pair is any of the
            // to-be-deleted indices. If so, delete the pair, i.e.
            // swap a different pair on its place from the back of the list.
            int32_t idx1 = pairs.i[i];
            int32_t idx2 = pairs.j[i];
            int32_t j = 0;
            while (j < d_num_to_be_deleted) {
                int32_t tbd = toBeDeleted[j];
                if (idx1 == tbd || idx2 == tbd) {
                    // Start from the back of pair list and go backwards until
                    // neither of the pairs is in the to-be-deleted list.
                    bool pair_found = false;
                    int32_t swap_idx = 0;
                    while (!pair_found) {
                        pair_found = true;
                        swap_idx = atomicAdd(&d_num_pairs, -1) - 1;
                        const int32_t swap1 = pairs.i[swap_idx];
                        const int32_t swap2 = pairs.j[swap_idx];
                        int32_t k = 0;
                        while (k < d_num_to_be_deleted) {
                            tbd = toBeDeleted[k];
                            if (swap1 == tbd || swap2 == tbd) {
                                pair_found = false;
                                break;
                            }
                            k += 1;
                        }
                    }
                    pairs.i[i] = pairs.i[swap_idx];
                    pairs.j[i] = pairs.j[swap_idx];
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
            while (j < d_num_to_be_deleted) {
                // The old, swapped indices were stored after the deleted
                // indices
                int32_t swapped = toBeDeleted[d_num_to_be_deleted + j];
                if (idx1 == swapped) {
                    pairs.i[i] = toBeDeleted[j];
                } else if (idx2 == swapped) {
                    pairs.j[i] = toBeDeleted[j];
                }

                j += 1;
            }
        }

        if (i < bcn) {
            bubbles.r[i] = bubbles.r[i] * vol_mul;
        }
    }
}

// ==============================================
// Miscellaneous
// ==============================================
__global__ void potentialEnergy(Bubbles bubbles, Pairs pairs, double *energy) {
    for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < d_num_pairs;
         i += gridDim.x * blockDim.x) {
        const int32_t idx1 = pairs.i[i];
        const int32_t idx2 = pairs.j[i];
        double e = bubbles.r[idx1] + bubbles.r[idx2] -
                   length(wrappedDifference(bubbles.x[idx1], bubbles.y[idx1],
                                            bubbles.z[idx1], bubbles.x[idx2],
                                            bubbles.y[idx2], bubbles.z[idx2]));

        if (e > 0) {
            e *= e;
            atomicAdd(&energy[idx1], e);
            atomicAdd(&energy[idx2], e);
        }
    }
}

__global__ void euler(double ts, Bubbles bubbles) {
    // Euler integration
    for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += blockDim.x * gridDim.x) {
        bubbles.xp[i] += bubbles.dxdtp[i] * ts;
        bubbles.yp[i] += bubbles.dydtp[i] * ts;
        if (d_constants->dimensionality == 3) {
            bubbles.zp[i] += bubbles.dzdtp[i] * ts;
        }
    }
}

__global__ void transformPositions(bool normalize, Bubbles bubbles) {
    const dvec lbb = d_constants->lbb;
    const dvec interval = d_constants->interval;
    for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += gridDim.x * blockDim.x) {
        if (normalize) {
            bubbles.x[i] = (bubbles.x[i] - lbb.x) / interval.x;
            bubbles.y[i] = (bubbles.y[i] - lbb.y) / interval.y;
            if (d_constants->dimensionality == 3) {
                bubbles.z[i] = (bubbles.z[i] - lbb.z) / interval.z;
            }
        } else {
            bubbles.x[i] = interval.x * bubbles.x[i] + lbb.x;
            bubbles.y[i] = interval.y * bubbles.y[i] + lbb.y;
            if (d_constants->dimensionality == 3) {
                bubbles.z[i] = interval.z * bubbles.z[i] + lbb.z;
            }
        }
    }
}

__global__ void wrapOverPeriodicBoundaries(Bubbles bubbles) {
    const dvec lbb = d_constants->lbb;
    const dvec tfr = d_constants->tfr;
    for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += gridDim.x * blockDim.x) {
        auto wrap = [&i](double *p, int32_t *wc, double x, double low,
                         double high) {
            int32_t mult = x < low ? 1 : (x > high ? -1 : 0);
            p[i] = x + (high - low) * (double)mult;
            wc[i] -= mult;
        };

        if (!d_constants->x_wall)
            wrap(bubbles.x, bubbles.wrap_count_x, bubbles.x[i], lbb.x, tfr.x);
        if (!d_constants->y_wall)
            wrap(bubbles.y, bubbles.wrap_count_y, bubbles.y[i], lbb.y, tfr.y);
        if (!d_constants->z_wall)
            wrap(bubbles.z, bubbles.wrap_count_z, bubbles.z[i], lbb.z, tfr.z);
    }
}

__global__ void calculateVolumes(Bubbles bubbles, double *volumes) {
    for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += gridDim.x * blockDim.x) {
        const double radius = bubbles.r[i];
        double volume = radius * radius * CUBBLE_PI;
        if (d_constants->dimensionality == 3) {
            volume *= radius * 1.33333333333333333333333333;
        }

        volumes[i] = volume;
    }
}

__global__ void assignDataToBubbles(uvec bubblesPerDim, double avgRad,
                                    Bubbles bubbles) {
    const dvec interval = d_constants->interval;
    const dvec lbb = d_constants->lbb;
    const dvec tfr = d_constants->tfr;
    const double min_rad = d_constants->min_rad;
    double *w = bubbles.rp;
    for (int32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < bubbles.count;
         i += gridDim.x * blockDim.x) {
        bubbles.index[i] = i;
        dvec pos(0, 0, 0);
        pos.x = (i % bubblesPerDim.x) / (double)bubblesPerDim.x;
        pos.y =
            ((i / bubblesPerDim.x) % bubblesPerDim.y) / (double)bubblesPerDim.y;

        dvec random_offset(bubbles.x[i], bubbles.y[i], 0);
        if (d_constants->dimensionality == 3) {
            random_offset.z = bubbles.z[i];
            pos.z = (i / (bubblesPerDim.x * bubblesPerDim.y)) /
                    (double)bubblesPerDim.z;
        }
        pos *= interval;
        random_offset = normalize(random_offset) * avgRad * w[i];
        pos += random_offset;

        double rad = bubbles.r[i];
        rad = ::abs(rad);
        rad = rad > min_rad ? rad : rad + min_rad;
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
        if (d_constants->dimensionality == 3) {
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
} // namespace cubble
