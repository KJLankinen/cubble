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

#pragma once

#include "vec.h"
#include <cstdint>

namespace cubble {

struct InputParameters;

struct ParticleBox {
    const dvec shape;
    const uvec particles_per_dimension;
    const uint32_t num_requested;
    const uint32_t num_particles;
    const uint32_t dimensionality;

    // Old way
    ParticleBox(const dvec &shape, uint32_t num_requested,
                uint32_t dimensionality);
    // New way
    ParticleBox(const InputParameters &ip);

  private:
    static uvec computeParticlesPerDimension(const dvec &shape,
                                             uint32_t num_requested,
                                             uint32_t dimensionality);
    };
}
