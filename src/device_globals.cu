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

namespace cubble {
__device__ Constants *d_constants;
__device__ double d_total_area;
__device__ double d_total_overlap_area;
__device__ double d_total_overlap_area_per_radius;
__device__ double d_total_area_per_radius;
__device__ double d_total_volume_new;
__device__ double d_max_radius;
__device__ bool d_error_encountered;
__device__ int32_t d_num_pairs;
__device__ int32_t d_num_pairs_new;
__device__ int32_t d_num_to_be_deleted;
}; // namespace cubble

