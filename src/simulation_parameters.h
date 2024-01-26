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
#include <nlohmann/json.hpp>

namespace cubble {
// Helper structs matching the input json structure
struct Bubble {
    double radius_mean = 0.0;
    double radius_std = 0.0;
    uint32_t num_start = 0;
    uint32_t num_end = 0;
};
ASSERT_SIZE(Bubble, 24);

struct Box {
    dvec relative_dimensions = dvec(1.0);
    uint8_t dimensionality = 0;
    uint8_t padding[7];
};
ASSERT_SIZE(Box, 32);

struct Wall {
    double drag = 0.0;
    bool x = false;
    bool y = false;
    bool z = false;
    uint8_t padding[5];
};
ASSERT_SIZE(Wall, 16);

struct Flow {
    dvec relative_dimensions = dvec(0.0);
    dvec left_bottom_back = dvec(0.0);
    dvec right_top_front = dvec(0.0);
    bool impose = false;
    uint8_t padding[7];
};
ASSERT_SIZE(Flow, 80);

struct Constants {
    double phi = 0.0;
    double mu = 0.0;
    double sigma = 0.0;
    double kappa = 0.0;
    double K = 0.0;
};
ASSERT_SIZE(Constants, 40);

struct SimulationParameters {
    Flow flow = {};
    Constants constants = {};
    Box box = {};
    Bubble bubble = {};
    Wall wall = {};
    // 192 + 32
    std::string comments_md = "";
    std::string snap_shot_filename = "bubbles";
    double error_tolerance = 0.0;
    double stabilization_max_delta_energy = 1e-5;
    double snap_shot_frequency = 0.0;
    uint32_t rng_seed = 426;
    uint32_t stabilization_steps = 1e4;
};
constexpr size_t bytes = 224 + 2 * sizeof(std::string);
ASSERT_SIZE(SimulationParameters, bytes);

void to_json(nlohmann::json &, const Bubble &);
void to_json(nlohmann::json &, const Box &);
void to_json(nlohmann::json &, const Wall &);
void to_json(nlohmann::json &, const Flow &);
void to_json(nlohmann::json &, const Constants &);
void to_json(nlohmann::json &, const SimulationParameters &);
void from_json(const nlohmann::json &, Bubble &);
void from_json(const nlohmann::json &, Box &);
void from_json(const nlohmann::json &, Wall &);
void from_json(const nlohmann::json &, Flow &);
void from_json(const nlohmann::json &, Constants &);
void from_json(const nlohmann::json &, SimulationParameters &);
} // namespace cubble
