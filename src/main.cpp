/*
    Cubble
    Copyright (C) 2019  Juhana Lankinen

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
#include "result.h"
#include <cstdint>

namespace cubble {
int32_t run(int32_t argc, char **argv);
}

int32_t main(int32_t argc, char **argv) {
    // TEMP
    {
        cubble::Result<cubble::Config> result = cubble::parse(argc, argv);
        printf("result: %d, with message %s\n", result.ok, result.msg.c_str());
    }
    return cubble::run(argc, argv);
}
