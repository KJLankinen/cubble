#include <iostream>
#include "include/json.hpp"
#include <fstream>

int main(int argc, char **argv)
{
    int success = EXIT_SUCCESS;

    nlohmann::json j;
    j["phi"] = 0.0;
    j["test"] = "blaa blaa";
    j["values"] = {1, 2, 3, 4, 5};
    j["obj"] = {{"first_thing", 16}, {"second_thing", "kissa"}};

    std::ofstream outputFile("test.json");
    if (outputFile)
	outputFile << std::setw(4) << j;

    if (outputFile.is_open())
	outputFile.close();

    std::cout << j << std::endl;
    
    return success;
}
