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
#include "parsing.h"
#include "simulation.h"
#include <cstdint>
#include <cstdlib>

namespace cubble {
int32_t run(int32_t argc, char **argv);
}

int32_t newPath(int32_t argc, char **argv) {
    auto [config, msg, ok, _] = cubble::parse(argc, argv);
    if (ok) {
        cubble::Simulation simulation(std::move(config));
        simulation.run();

        return EXIT_SUCCESS;
    } else {
        printf("Input parsing failed: %s\n", msg.c_str());
        return EXIT_FAILURE;
    }
}

int32_t oldPath(int32_t argc, char **argv) { return cubble::run(argc, argv); }

int32_t main(int32_t argc, char **argv) { oldPath(argc, argv); }
