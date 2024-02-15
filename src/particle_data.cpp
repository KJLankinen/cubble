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

#include "particle_data.h"
#include "constants.h"
#include "particle_box.h"
#include <random>

namespace cubble {
ParticleData::ParticleData(const ParticleBox &box, const dvec &rtf,
                           const dvec &lbb, double mean, double standard_dev,
                           uint32_t rng_seed)
    : x(generateCoordinate(box, rtf, lbb, Coordinate::x)),
      y(generateCoordinate(box, rtf, lbb, Coordinate::y)),
      z(generateCoordinate(box, rtf, lbb, Coordinate::z)),
      r(getRadii(box.num_particles, mean, standard_dev, rng_seed)) {}

const std::vector<double> &ParticleData::getRadii(size_t num, double mean,
                                                  double standard_dev,
                                                  uint32_t rng_seed) {
    const static std::vector<double> r([&num, &mean, &standard_dev,
                                        &rng_seed]() {
        std::random_device rd{};
        std::mt19937 generator{rd()};
        generator.seed(rng_seed);
        std::normal_distribution distribution{mean, standard_dev};
        const double min_radius = ConstantsNew::getMinRad(mean);

        std::vector<double> r(num);
        std::generate(
            r.begin(), r.end(), [&distribution, &generator, &min_radius]() {
                return std::max(std::abs(distribution(generator)), min_radius);
            });

        return r;
    }());

    return r;
}

std::vector<double> ParticleData::generateCoordinate(const ParticleBox &box,
                                                     const dvec &rtf,
                                                     const dvec &lbb,
                                                     Coordinate coord) {
    const size_t ppdx = box.particles_per_dimension.x;
    const size_t ppdy = box.particles_per_dimension.y;
    const size_t ppdz = box.particles_per_dimension.z;
    const uint32_t dim = box.dimensionality;
    const dvec interval = rtf - lbb;

    std::vector<double> data(box.num_particles);
    std::generate(
        data.begin(), data.end(),
        [&ppdx, &ppdy, &ppdz, &dim, &lbb, &interval, coord, i = 0ul]() mutable {
            auto gen_x = [&ppdx](size_t i) {
                return (static_cast<double>(i % ppdx) + 0.5) /
                       static_cast<double>(ppdx);
            };

            auto gen_y = [&ppdx, &ppdy](size_t i) {
                return (static_cast<double>((i / ppdx) % ppdy) + 0.5) /
                       static_cast<double>(ppdy);
            };

            auto gen_z =
                [&ppdx, &ppdy, &ppdz, &dim](size_t i) {
                    return dim == 3 ? (static_cast<double>((i / (ppdx * ppdy)) %
                                                           ppdz) +
                                       0.5) /
                                          static_cast<double>(ppdz)
                                    : 0;
                };

            double value = 0.0;
            switch (coord) {
            case Coordinate::x: {
                value = gen_x(i) * interval.x + lbb.x;
                break;
            }
            case Coordinate::y: {
                value = gen_y(i) * interval.y + lbb.y;
                break;
            }
            case Coordinate::z: {
                value = gen_z(i) * interval.z + lbb.z;
                break;
            }
            }

            i++;
            return value;
        });

    return data;
    }
}

