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

#include "parsing.h"
#include "config.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace cubble {
Result<Config> parse(std::string inputFilename, std::string outputFilename) {
    std::stringstream errmsg;
    std::fstream file(inputFilename, std::ios::in);
    if (file.is_open()) {
        nlohmann::json j;
        try {
            file >> j;
            return Result<Config>::Ok(
                Config{j.template get<InputParameters>(), outputFilename});
        } catch (const nlohmann::json::exception &e) {
            errmsg << "Error parsing input with filename '" << inputFilename
                   << "'. Exception thrown: " << e.what();
        }
    } else {
        errmsg << "Error opening input filename with name '" << inputFilename
               << "'";
    }

    return Result<Config>::Err(errmsg.str());
}

Result<Config> parse(int32_t argc, char **argv) {
    if (argc != 3) {
        std::stringstream ss;
        ss << "Incorrect number of command line arguments. Was expecting 2, "
              "got "
           << argc - 1 << "\nUsage: '" << argv[0]
           << " input_file output_file', where\n\tinput_file is the name of "
              "the .json file containing the simulation input\n\toutput_file "
              "is the name of the .json file to where results are output";

        return Result<Config>::Err(ss.str());
    }

    return parse(argv[1], argv[2]);
}
} // namespace cubble
