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

#include "constants.h"
#include "particle_box.h"
#include "particle_data.h"
#include "vec.h"
#include <algorithm>

namespace cubble {
ConstantsNew::ConstantsNew(const InputParameters &in_params,
                           const ParticleBox &box,
                           const std::vector<double> &radii)
    : lbb(0.0, 0.0, 0.0), rtf(computeRtf(in_params, box, radii)),
      interval(lbb - rtf),
      flow_lbb(interval * in_params.flow.left_bottom_back + lbb),
      flow_rtf(interval * in_params.flow.right_top_front + lbb),
      flow_vel(in_params.flow.velocity *
               ConstantsNew::computeFZeroPerMuZero(in_params)),
      average_surface_area_in(std::for_each(radii.begin(), radii.end(),
                                            Area(box.dimensionality == 3))
                                  .total_area /
                              box.num_particles),
      min_rad(getMinRad(in_params.bubble.radius_mean)),
      f_zero_per_mu_zero(ConstantsNew::computeFZeroPerMuZero(in_params)),
      k_parameter(in_params.input_constants.k),
      kappa(in_params.input_constants.kappa),
      wall_drag_strength(in_params.wall.drag),
      skin_radius(in_params.bubble.radius_mean * skin_radius_multiplier),
      bubble_volume_multiplier(computeVolumeMultiplier(lbb, rtf, in_params)),
      dimensionality(in_params.box.dimensionality), x_wall(in_params.wall.x),
      y_wall(in_params.wall.y), z_wall(in_params.wall.z), padding{0} {}

double ConstantsNew::getMinRad(double radius_mean) {
    return ConstantsNew::min_radius_multiplier * radius_mean;
}

double ConstantsNew::computeFZeroPerMuZero(const InputParameters &in_params) {
    return in_params.input_constants.sigma * in_params.bubble.radius_mean /
           in_params.input_constants.mu;
}

dvec ConstantsNew::computeRtf(const InputParameters &in_params,
                              const ParticleBox &box,
                              const std::vector<double> &radii) {
    dvec rel_dim = in_params.box.relative_dimensions;
    double t = std::for_each(radii.begin(), radii.end(),
                             Volume(box.dimensionality == 3))
                   .total_volume /
               (in_params.input_constants.phi * rel_dim.x * rel_dim.y);
    if (in_params.box.dimensionality == 3) {
        t /= rel_dim.z;
        t = std::cbrt(t);
    } else {
        t = std::sqrt(t);
        rel_dim.z = 0.0;
    }

    return dvec(t) * rel_dim;
}

double ConstantsNew::computeVolumeMultiplier(const dvec &lbb, const dvec &rtf,
                                             const InputParameters &in_params) {
    const bool is_3d = in_params.box.dimensionality == 3;
    const dvec interval = rtf - lbb;

    const double volume = interval.x * interval.y;
    double mult = in_params.input_constants.phi * volume / CUBBLE_PI;
    if (is_3d) {
        mult = std::cbrt(0.75 * mult * interval.z);
    } else {
        mult = std::sqrt(mult);
    }

    return mult;
}

void to_json(nlohmann::json &j, const ConstantsNew &from) {
    j = nlohmann::json{
        {"lbb", from.lbb},
        {"rtf", from.rtf},
        {"interval", from.interval},
        {"flow_lbb", from.flow_lbb},
        {"flow_rtf", from.flow_rtf},
        {"flow_vel", from.flow_vel},
        {"average_surface_area_in", from.average_surface_area_in},
        {"min_rad", from.min_rad},
        {"f_zero_per_mu_zero", from.f_zero_per_mu_zero},
        {"k_parameter", from.k_parameter},
        {"kappa", from.kappa},
        {"wall_drag_strength", from.wall_drag_strength},
        {"skin_radius", from.skin_radius},
        {"bubble_volume_multiplier", from.bubble_volume_multiplier},
        {"dimensionality", from.dimensionality},
        {"x_wall", from.x_wall},
        {"y_wall", from.y_wall},
        {"z_wall", from.z_wall}};
}
} // namespace cubble
