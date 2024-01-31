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

#include "config.h"

namespace cubble {
    void to_json(nlohmann::json &j, const Config &from) {
        j = nlohmann::json{{"output_filename", from.output_filename},
                           {"simulation_parameters", from.input_parameters}};
    }

    void from_json(const nlohmann::json &j, Config &to) {
        j.at("output_filename").get_to(to.output_filename);
        j.at("simulation_parameters").get_to(to.input_parameters);
    }
}
