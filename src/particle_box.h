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
#include <cmath>
#include <cstdint>

namespace cubble {
    struct ParticleBox {
        const dvec shape;
        const uvec particles_per_dimension;
        const uint32_t num_requested;
        const uint32_t num_particles;
        const uint32_t dimensionality;

        ParticleBox(const dvec &shape, uint32_t num_requested,
                    uint32_t dimensionality)
            : shape(shape),
              particles_per_dimension(computeParticlesPerDimension(
                  shape, num_requested, dimensionality)),
              num_requested(num_requested),
              num_particles(
                  particles_per_dimension.x * particles_per_dimension.y *
                  (particles_per_dimension.z > 0 ? particles_per_dimension.z
                                                 : 1)),
              dimensionality(dimensionality) {}

      private:
        static uvec computeParticlesPerDimension(const dvec &shape,
                                                 uint32_t num_requested,
                                                 uint32_t dimensionality) {
            uvec particles_per_dim;
            if (dimensionality == 3) {
                const double n = std::cbrt(num_requested);
                const double a = std::cbrt(shape.x / shape.y);
                const double b = std::cbrt(shape.x / shape.z);
                const double c = std::cbrt(shape.y / shape.z);
                particles_per_dim.x = (uint32_t)std::ceil(n * a * b);
                particles_per_dim.y = (uint32_t)std::ceil(n * c / a);
                particles_per_dim.z = (uint32_t)std::ceil(n / (b * c));
            } else {
                const double n = std::sqrt(num_requested);
                const double a = std::sqrt(shape.x / shape.y);
                particles_per_dim.x = (uint32_t)std::ceil(n * a);
                particles_per_dim.y = (uint32_t)std::ceil(n / a);
                particles_per_dim.z = 0U;
            }

            return particles_per_dim;
        }
    };
}
