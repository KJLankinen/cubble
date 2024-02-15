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

#include "input_parameters.h"
#include "vec.h"
#include <cstdint>
#include <nlohmann/json.hpp>

namespace {
// Helpers for computing variables from a vector of radii
struct Area {
    Area(bool is_3d) : is_3d(is_3d) {}
    void operator()(double r) {
        const double rpi = CUBBLE_PI * r;
        double area = 2.0 * rpi;
        if (is_3d) {
            area *= 2.0 * r;
        }
        total_area += area;
    }
    double total_area = 0.0;
    bool is_3d = false;
    uint8_t padding[7];
};

struct Volume {
    Volume(bool is_3d) : is_3d(is_3d) {}
    void operator()(double r) {
        const double rpi = CUBBLE_PI * r;
        double volume = r * rpi;
        if (is_3d) {
            volume *= 4.0 / 3.0 * r;
        }
        total_volume += volume;
    }
    double total_volume = 0.0;
    bool is_3d = false;
    uint8_t padding[7];
};
} // namespace

namespace cubble {
struct ParticleBox;

struct ConstantsNew {
    constexpr static double skin_radius_multiplier = 0.3;
    constexpr static double min_radius_multiplier = 0.1;

    const dvec lbb;
    const dvec rtf;
    const dvec interval;

    const dvec flow_lbb;
    const dvec flow_rtf;
    const dvec flow_vel;

    const double average_surface_area_in;
    const double min_rad;
    const double f_zero_per_mu_zero;
    const double k_parameter;
    const double kappa;
    const double wall_drag_strength;
    const double skin_radius;
    const double bubble_volume_multiplier;

    const uint32_t dimensionality;

    const bool x_wall;
    const bool y_wall;
    const bool z_wall;

    const uint8_t padding[1];

    ConstantsNew(const InputParameters &in_params, const ParticleBox &box,
                 const std::vector<double> &radii);
    static double getMinRad(double radius_mean);

  private:
    static double computeFZeroPerMuZero(const InputParameters &in_params);
    static dvec computeRtf(const InputParameters &in_params,
                           const ParticleBox &box,
                           const std::vector<double> &radii);
    static double computeVolumeMultiplier(const dvec &lbb, const dvec &rtf,
                                          const InputParameters &in_params);
};
ASSERT_SIZE(ConstantsNew, 216);

void to_json(nlohmann::json &, const ConstantsNew &);
}
