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
#include "free_functions.cuh"

namespace cubble {
__device__ double correct(int32_t i, double ts, double *pp, double *p,
                          double *v, double *vp, double *old, double *maxErr) {
    const double predicted = pp[i];
    const double corrected = p[i] + 0.5 * ts * (v[i] + vp[i]);
    pp[i] = corrected;
    maxErr[threadIdx.x] =
        fmax(maxErr[threadIdx.x], ::abs(predicted - corrected));

    return corrected - old[i];
}

__device__ void addFlowVelocity(Bubbles &bubbles, int32_t i) {
    double multiplier = bubbles.num_neighbors[i];
    multiplier = (multiplier > 0 ? 1.0 / multiplier : 0.0);
    const double xi = bubbles.xp[i];
    const double yi = bubbles.yp[i];
    double ri_sq = bubbles.rp[i];
    ri_sq *= ri_sq;

    int32_t inside = (int32_t)((xi < d_constants->flow_tfr.x &&
                                xi > d_constants->flow_lbb.x) ||
                               ((d_constants->flow_lbb.x - xi) *
                                    (d_constants->flow_lbb.x - xi) <=
                                ri_sq) ||
                               ((d_constants->flow_tfr.x - xi) *
                                    (d_constants->flow_tfr.x - xi) <=
                                ri_sq));

    inside *= (int32_t)((yi < d_constants->flow_tfr.y &&
                         yi > d_constants->flow_lbb.y) ||
                        ((d_constants->flow_lbb.y - yi) *
                             (d_constants->flow_lbb.y - yi) <=
                         ri_sq) ||
                        ((d_constants->flow_tfr.y - yi) *
                             (d_constants->flow_tfr.y - yi) <=
                         ri_sq));

    if (d_constants->dimensionality == 3) {
        const double zi = bubbles.zp[i];
        inside *= (int32_t)((zi < d_constants->flow_tfr.z &&
                             zi > d_constants->flow_lbb.z) ||
                            ((d_constants->flow_lbb.z - zi) *
                                 (d_constants->flow_lbb.z - zi) <=
                             ri_sq) ||
                            ((d_constants->flow_tfr.z - zi) *
                                 (d_constants->flow_tfr.z - zi) <=
                             ri_sq));

        bubbles.dzdtp[i] += !inside * multiplier * bubbles.flow_vz[i] +
                            d_constants->flow_vel.z * inside;
    }

    // Either add the average velocity of neighbors or the imposed flow,
    // if the bubble is inside the flow area
    bubbles.dxdtp[i] += !inside * multiplier * bubbles.flow_vx[i] +
                        d_constants->flow_vel.x * inside;
    bubbles.dydtp[i] += !inside * multiplier * bubbles.flow_vy[i] +
                        d_constants->flow_vel.y * inside;
}

__device__ void addWallVelocity(Bubbles &bubbles, int32_t i) {
    const double drag = 1.0 - d_constants->wall_drag_strength;
    const double rad = bubbles.rp[i];
    double x_drag = 1.0;
    double y_drag = 1.0;
    double velocity = 0.0;

    auto touches_wall = [&i, &rad, &velocity](double x, double low,
                                              double high) -> bool {
        double d1 = x - low;
        double d2 = x - high;
        d1 = d1 * d1 < d2 * d2 ? d1 : d2;
        if (rad * rad >= d1 * d1) {
            d2 = d1 < 0.0 ? -1.0 : 1.0;
            d1 *= d2;
            velocity = d2 * d_constants->f_zero_per_mu_zero * (1.0 - d1 / rad);
            return true;
        }

        return false;
    };

    velocity = 0.0;
    if (d_constants->x_wall &&
        touches_wall(bubbles.xp[i], d_constants->lbb.x, d_constants->tfr.x)) {
        bubbles.dxdtp[i] += velocity;
        bubbles.dydtp[i] *= drag;
        bubbles.dzdtp[i] *= drag;
        x_drag = drag;
    }

    velocity = 0.0;
    if (d_constants->y_wall &&
        touches_wall(bubbles.yp[i], d_constants->lbb.y, d_constants->tfr.y)) {
        bubbles.dxdtp[i] *= drag;
        bubbles.dydtp[i] += velocity * x_drag;
        bubbles.dzdtp[i] *= drag;
        y_drag = drag;
    }

    velocity = 0.0;
    if (d_constants->z_wall &&
        touches_wall(bubbles.zp[i], d_constants->lbb.z, d_constants->tfr.z)) {
        bubbles.dxdtp[i] *= drag;
        bubbles.dydtp[i] *= drag;
        bubbles.dzdtp[i] += velocity * x_drag * y_drag;
    }
}

__device__ void comparePair(int32_t idx1, int32_t idx2, int32_t *histogram,
                            int32_t *pairI, int32_t *pairJ, Bubbles &bubbles) {
    const double max_distance =
        bubbles.r[idx1] + bubbles.r[idx2] + d_constants->skin_radius;
    if (lengthSq(wrappedDifference(
            bubbles.x[idx1], bubbles.y[idx1], bubbles.z[idx1], bubbles.x[idx2],
            bubbles.y[idx2], bubbles.z[idx2])) < max_distance * max_distance) {
        // Set the smaller idx to idx1 and larger to idx2
        int32_t id = idx1 > idx2 ? idx1 : idx2;
        idx1 = idx1 < idx2 ? idx1 : idx2;
        idx2 = id;

        atomicAdd(&histogram[idx1], 1);
        id = atomicAdd(&d_num_pairs, 1);
        pairI[id] = idx1;
        pairJ[id] = idx2;
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

        d_error_encountered = true;
    }
}

__device__ dvec wrappedDifference(double x1, double y1, double z1, double x2,
                                  double y2, double z2) {
    dvec d1 = dvec(x1 - x2, y1 - y2, 0.0);
    if (3 == d_constants->dimensionality) {
        d1.z = z1 - z2;
    }
    dvec d2 = d1;
    dvec temp = d_constants->interval - abs(d1);
    if (!d_constants->x_wall && temp.x * temp.x < d1.x * d1.x) {
        d2.x = temp.x * (d1.x < 0 ? 1.0 : -1.0);
    }
    if (!d_constants->y_wall && temp.y * temp.y < d1.y * d1.y) {
        d2.y = temp.y * (d1.y < 0 ? 1.0 : -1.0);
    }
    if (3 == d_constants->dimensionality && !d_constants->z_wall &&
        temp.z * temp.z < d1.z * d1.z) {
        d2.z = temp.z * (d1.z < 0 ? 1.0 : -1.0);
    }

    return d2;
}

__device__ int32_t getNeighborCellIndex(int32_t cellIdx, ivec dim,
                                        int32_t neighborNum) {
    ivec idx_vec = get3DIdxFrom1DIdx(cellIdx, dim);
    switch (neighborNum) {
    case 0:
        // self
        break;
    case 1:
        idx_vec += ivec(-1, 1, 0);
        break;
    case 2:
        idx_vec += ivec(-1, 0, 0);
        break;
    case 3:
        idx_vec += ivec(-1, -1, 0);
        break;
    case 4:
        idx_vec += ivec(0, -1, 0);
        break;
    case 5:
        idx_vec += ivec(-1, 1, -1);
        break;
    case 6:
        idx_vec += ivec(-1, 0, -1);
        break;
    case 7:
        idx_vec += ivec(-1, -1, -1);
        break;
    case 8:
        idx_vec += ivec(0, 1, -1);
        break;
    case 9:
        idx_vec += ivec(0, 0, -1);
        break;
    case 10:
        idx_vec += ivec(0, -1, -1);
        break;
    case 11:
        idx_vec += ivec(1, 1, -1);
        break;
    case 12:
        idx_vec += ivec(1, 0, -1);
        break;
    case 13:
        idx_vec += ivec(1, -1, -1);
        break;
    default:
        printf("Should never end up here!\n");
        break;
    }

    if (!d_constants->x_wall) {
        idx_vec.x += dim.x;
        idx_vec.x %= dim.x;
    } else if (idx_vec.x < 0 || idx_vec.x >= dim.x) {
        return -1;
    }

    if (!d_constants->y_wall) {
        idx_vec.y += dim.y;
        idx_vec.y %= dim.y;
    } else if (idx_vec.y < 0 || idx_vec.y >= dim.y) {
        return -1;
    }

    if (!d_constants->z_wall) {
        idx_vec.z += dim.z;
        idx_vec.z %= dim.z;
    } else if (idx_vec.z < 0 || idx_vec.z >= dim.z) {
        return -1;
    }

    return get1DIdxFrom3DIdx(idx_vec, dim);
}

__device__ int32_t getCellIdxFromPos(double x, double y, double z,
                                     ivec cellDim) {
    const dvec lbb = d_constants->lbb;
    const dvec interval = d_constants->interval;
    const int32_t xid = ::floor(cellDim.x * (x - lbb.x) / interval.x);
    const int32_t yid = ::floor(cellDim.y * (y - lbb.y) / interval.y);
    int32_t zid = 0;
    if (d_constants->dimensionality == 3) {
        zid = ::floor(cellDim.z * (z - lbb.z) / interval.z);
    }

    return get1DIdxFrom3DIdx(ivec(xid, yid, zid), cellDim);
}

__device__ int32_t get1DIdxFrom3DIdx(ivec idxVec, ivec cellDim) {
    // Linear encoding
    return idxVec.z * cellDim.x * cellDim.y + idxVec.y * cellDim.x + idxVec.x;
}

__device__ ivec get3DIdxFrom1DIdx(int32_t idx, ivec cellDim) {
    ivec idx_vec(0, 0, 0);
    // Linear decoding
    idx_vec.x = idx % cellDim.x;
    idx_vec.y = (idx / cellDim.x) % cellDim.y;
    if (d_constants->dimensionality == 3) {
        idx_vec.z = idx / (cellDim.x * cellDim.y);
    }

    return idx_vec;
}

__device__ void resetDeviceGlobals() {
    d_total_area = 0.0;
    d_total_overlap_area = 0.0;
    d_total_overlap_area_per_radius = 0.0;
    d_total_area_per_radius = 0.0;
    d_total_volume_new = 0.0;
    d_max_radius = 0.0;
    d_num_to_be_deleted = 0;
    d_num_pairs_new = d_num_pairs;
    d_error_encountered = false;
}
} // namespace cubble
