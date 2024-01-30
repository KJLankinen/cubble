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

#include "simulation_parameters.h"

namespace cubble {
void to_json(nlohmann::json &j, const Bubble &from) {
    j = nlohmann::json{{"num_start", from.num_start},
                       {"num_end", from.num_end},
                       {"radius_mean", from.radius_mean},
                       {"radius_std", from.radius_std}};
}

void to_json(nlohmann::json &j, const Box &from) {
    j = nlohmann::json{{"dimensionality", from.dimensionality},
                       {"relative_dimensions", from.relative_dimensions}};
}

void to_json(nlohmann::json &j, const Wall &from) {
    j = nlohmann::json{
        {"drag", from.drag}, {"x", from.x}, {"y", from.y}, {"z", from.z}};
}

void to_json(nlohmann::json &j, const Flow &from) {
    j = nlohmann::json{{"impose", from.impose},
                       {"velocity", from.velocity},
                       {"left_bottom_back", from.left_bottom_back},
                       {"right_top_front", from.right_top_front}};
}

void to_json(nlohmann::json &j, const Constants &from) {
    j = nlohmann::json{{"phi", from.phi},
                       {"mu", from.mu},
                       {"sigma", from.sigma},
                       {"kappa", from.kappa},
                       {"k", from.k}};
}

void to_json(nlohmann::json &j, const SimulationParameters &from) {
    j = nlohmann::json{
        {"comments_md", from.comments_md},
        {"bubble", from.bubble},
        {"box", from.box},
        {"wall", from.wall},
        {"flow", from.flow},
        {"constants", from.constants},
        {"error_tolerance", from.error_tolerance},
        {"rng_seed", from.rng_seed},
        {"stabilization_steps", from.stabilization_steps},
        {"stabilization_max_delta_energy", from.stabilization_max_delta_energy},
        {"snap_shot_filename", from.snap_shot_filename},
        {"snap_shot_frequency", from.snap_shot_frequency}};
}

void from_json(const nlohmann::json &j, Bubble &to) {
    j.at("num_start").get_to(to.num_start);
    j.at("num_end").get_to(to.num_end);
    j.at("radius_mean").get_to(to.radius_mean);
    j.at("radius_std").get_to(to.radius_std);
}

void from_json(const nlohmann::json &j, Box &to) {
    j.at("dimensionality").get_to(to.dimensionality);
    j.at("relative_dimensions").get_to(to.relative_dimensions);
}

void from_json(const nlohmann::json &j, Wall &to) {
    j.at("drag").get_to(to.drag);
    j.at("x").get_to(to.x);
    j.at("y").get_to(to.y);
    j.at("z").get_to(to.z);
}

void from_json(const nlohmann::json &j, Flow &to) {
    j.at("impose").get_to(to.impose);
    j.at("velocity").get_to(to.velocity);
    j.at("left_bottom_back").get_to(to.left_bottom_back);
    j.at("right_top_front").get_to(to.right_top_front);
}

void from_json(const nlohmann::json &j, Constants &to) {
    j.at("phi").get_to(to.phi);
    j.at("mu").get_to(to.mu);
    j.at("sigma").get_to(to.sigma);
    j.at("kappa").get_to(to.kappa);
    j.at("k").get_to(to.k);
}

void from_json(const nlohmann::json &j, SimulationParameters &to) {
    j.at("comments_md").get_to(to.comments_md);
    j.at("bubble").get_to(to.bubble);
    j.at("box").get_to(to.box);
    j.at("wall").get_to(to.wall);
    j.at("flow").get_to(to.flow);
    j.at("constants").get_to(to.constants);
    j.at("error_tolerance").get_to(to.error_tolerance);
    j.at("rng_seed").get_to(to.rng_seed);
    j.at("stabilization_steps").get_to(to.stabilization_steps);
    j.at("stabilization_max_delta_energy")
        .get_to(to.stabilization_max_delta_energy);
    j.at("snap_shot_filename").get_to(to.snap_shot_filename);
    j.at("snap_shot_frequency").get_to(to.snap_shot_frequency);
}
} // namespace cubble
