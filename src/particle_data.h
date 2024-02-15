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
#include <vector>

namespace cubble {

struct ParticleBox;
struct InputParameters;
struct RadiusData;

struct ParticleData {
  private:
    enum class Coordinate {
        x,
        y,
        z,
    };

  public:
    const std::vector<double> x;
    const std::vector<double> y;
    const std::vector<double> z;
    const std::vector<double> &r;

    ParticleData(const ParticleBox &box, const dvec &rtf, const dvec &lbb,
                 double mean, double standard_dev, uint32_t rng_seed);

    static const std::vector<double> &
    getRadii(size_t num, double mean, double standard_dev, uint32_t rng_seed);

  private:
    static std::vector<double> generateCoordinate(const ParticleBox &box,
                                                  const dvec &rtf,
                                                  const dvec &lbb,
                                                  Coordinate coord);
};

void to_json(nlohmann::json &, const ParticleData &);
} // namespace cubble
