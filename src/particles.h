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

#include "particle_box.h"
#include <array>
#include <cstddef>
#include <cstdlib>
#include <random>
#include <vector>

namespace cubble {
struct ParticleData {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> r;
    double max_radius;
    double average_input_area;

    ParticleData(const ParticleBox &box, double radius_mean, double radius_std,
                 double min_radius, uint32_t seed)
        : x(box.num_particles), y(box.num_particles), z(box.num_particles),
          r(box.num_particles), max_radius(0.0), average_input_area(0.0) {
        const size_t ppdx = box.particles_per_dimension.x;
        const size_t ppdy = box.particles_per_dimension.y;
        const size_t ppdz = box.particles_per_dimension.z;
        const uint32_t dim = box.dimensionality;

        std::random_device rd{};
        std::mt19937 generator{rd()};
        generator.seed(seed);
        std::normal_distribution distribution{radius_mean, radius_std};

        auto gen_x = [&ppdx](size_t i) {
            return (static_cast<double>(i % ppdx) + 0.5) /
                   static_cast<double>(ppdx);
        };

        auto gen_y = [&ppdx, &ppdy](size_t i) {
            return (static_cast<double>((i / ppdx) % ppdy) + 0.5) /
                   static_cast<double>(ppdy);
        };

        auto gen_z = [&ppdx, &ppdy, &ppdz, &dim](size_t i) {
            return dim == 3 ? (static_cast<double>((i / (ppdx * ppdy)) % ppdz) +
                               0.5) /
                                  static_cast<double>(ppdz)
                            : 0;
        };

        auto gen_r = [&distribution, &generator, &min_radius]() {
            const double r = std::abs(distribution(generator));
            return r > min_radius ? r : r + min_radius;
        };

        for (size_t i = 0; i < box.num_particles; i++) {
            x[i] = gen_x(i);
            y[i] = gen_y(i);
            z[i] = gen_z(i);
            const double new_r = gen_r();
            r[i] = new_r;

            max_radius = max_radius > new_r ? max_radius : new_r;

            double area = 2.0 * CUBBLE_PI * new_r;
            if (box.dimensionality == 3) {
                area *= 2.0 * new_r;
            }
            average_input_area += area / box.num_particles;
        }
    }
};

} // namespace cubble
