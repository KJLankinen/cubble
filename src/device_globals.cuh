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

#pragma once

#include "data_definitions.h"

namespace cubble {
extern __device__ Constants *d_constants;
extern __device__ double d_total_area;
extern __device__ double d_total_overlap_area;
extern __device__ double d_total_overlap_area_per_radius;
extern __device__ double d_total_area_per_radius;
extern __device__ double d_total_volume_new;
extern __device__ double d_max_radius;
extern __device__ bool d_error_encountered;
extern __device__ int32_t d_num_pairs;
extern __device__ int32_t d_num_pairs_new;
extern __device__ int32_t d_num_to_be_deleted;
}; // namespace cubble
