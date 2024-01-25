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
    double multiplier = bubbles.numNeighbors[i];
    multiplier = (multiplier > 0 ? 1.0 / multiplier : 0.0);
    const double xi = bubbles.xp[i];
    const double yi = bubbles.yp[i];
    double riSq = bubbles.rp[i];
    riSq *= riSq;

    int32_t inside =
        (int32_t)((xi < dConstants->flowTfr.x && xi > dConstants->flowLbb.x) ||
                  ((dConstants->flowLbb.x - xi) *
                       (dConstants->flowLbb.x - xi) <=
                   riSq) ||
                  ((dConstants->flowTfr.x - xi) *
                       (dConstants->flowTfr.x - xi) <=
                   riSq));

    inside *=
        (int32_t)((yi < dConstants->flowTfr.y && yi > dConstants->flowLbb.y) ||
                  ((dConstants->flowLbb.y - yi) *
                       (dConstants->flowLbb.y - yi) <=
                   riSq) ||
                  ((dConstants->flowTfr.y - yi) *
                       (dConstants->flowTfr.y - yi) <=
                   riSq));

    if (dConstants->dimensionality == 3) {
        const double zi = bubbles.zp[i];
        inside *= (int32_t)((zi < dConstants->flowTfr.z &&
                             zi > dConstants->flowLbb.z) ||
                            ((dConstants->flowLbb.z - zi) *
                                 (dConstants->flowLbb.z - zi) <=
                             riSq) ||
                            ((dConstants->flowTfr.z - zi) *
                                 (dConstants->flowTfr.z - zi) <=
                             riSq));

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

__device__ void addWallVelocity(Bubbles &bubbles, int32_t i) {
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
    if (dConstants->xWall &&
        touchesWall(bubbles.xp[i], dConstants->lbb.x, dConstants->tfr.x)) {
        bubbles.dxdtp[i] += velocity;
        bubbles.dydtp[i] *= drag;
        bubbles.dzdtp[i] *= drag;
        xDrag = drag;
    }

    velocity = 0.0;
    if (dConstants->yWall &&
        touchesWall(bubbles.yp[i], dConstants->lbb.y, dConstants->tfr.y)) {
        bubbles.dxdtp[i] *= drag;
        bubbles.dydtp[i] += velocity * xDrag;
        bubbles.dzdtp[i] *= drag;
        yDrag = drag;
    }

    velocity = 0.0;
    if (dConstants->zWall &&
        touchesWall(bubbles.zp[i], dConstants->lbb.z, dConstants->tfr.z)) {
        bubbles.dxdtp[i] *= drag;
        bubbles.dydtp[i] *= drag;
        bubbles.dzdtp[i] += velocity * xDrag * yDrag;
    }
}

__device__ void comparePair(int32_t idx1, int32_t idx2, int32_t *histogram,
                            int32_t *pairI, int32_t *pairJ, Bubbles &bubbles) {
    const double maxDistance =
        bubbles.r[idx1] + bubbles.r[idx2] + dConstants->skinRadius;
    if (lengthSq(wrappedDifference(
            bubbles.x[idx1], bubbles.y[idx1], bubbles.z[idx1], bubbles.x[idx2],
            bubbles.y[idx2], bubbles.z[idx2])) < maxDistance * maxDistance) {
        // Set the smaller idx to idx1 and larger to idx2
        int32_t id = idx1 > idx2 ? idx1 : idx2;
        idx1 = idx1 < idx2 ? idx1 : idx2;
        idx2 = id;

        atomicAdd(&histogram[idx1], 1);
        id = atomicAdd(&dNumPairs, 1);
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
    dvec temp = dConstants->interval - abs(d1);
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

__device__ int32_t getNeighborCellIndex(int32_t cellIdx, ivec dim,
                                        int32_t neighborNum) {
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

__device__ int32_t getCellIdxFromPos(double x, double y, double z,
                                     ivec cellDim) {
    const dvec lbb = dConstants->lbb;
    const dvec interval = dConstants->interval;
    const int32_t xid = ::floor(cellDim.x * (x - lbb.x) / interval.x);
    const int32_t yid = ::floor(cellDim.y * (y - lbb.y) / interval.y);
    int32_t zid = 0;
    if (dConstants->dimensionality == 3) {
        zid = ::floor(cellDim.z * (z - lbb.z) / interval.z);
    }

    return get1DIdxFrom3DIdx(ivec(xid, yid, zid), cellDim);
}

__device__ int32_t get1DIdxFrom3DIdx(ivec idxVec, ivec cellDim) {
    // Linear encoding
    return idxVec.z * cellDim.x * cellDim.y + idxVec.y * cellDim.x + idxVec.x;
}

__device__ ivec get3DIdxFrom1DIdx(int32_t idx, ivec cellDim) {
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
    dMaxRadius = 0.0;
    dNumToBeDeleted = 0;
    dNumPairsNew = dNumPairs;
    dErrorEncountered = false;
}
} // namespace cubble
