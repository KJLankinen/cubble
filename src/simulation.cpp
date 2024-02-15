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

#include "simulation.h"
#include "particle_data.h"
#include "util.h"

namespace cubble {
Simulation::Simulation(Config &&config)
    : config(std::move(config)), p_box(config.input_parameters),
      constants(
          config.input_parameters, p_box,
          ParticleData::getRadii(p_box.num_particles,
                                 config.input_parameters.bubble.radius_mean,
                                 config.input_parameters.bubble.radius_std,
                                 config.input_parameters.rng_seed)) {
    cubble::print(config);
    // const cubble::ParticleData p_data(p_box, constants.rtf, constants.lbb);
}

void Simulation::run() {
    // TODO
}
} // namespace cubble
