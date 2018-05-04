#include <iostream>
#include <exception>
#include <stdexcept>
#include <sstream>

#include "Util.h"
#include "IOUtils.h"
#include "include/json.hpp"

using json = nlohmann::json;

void handleExceptions(const std::exception_ptr pExc)
{
    try
    {
	if (pExc)
	    std::rethrow_exception(pExc);
    }
    catch (const json::exception &e)
    {
	std::cout << "Encountered a json parse error."
		  << "\nMake sure the .json file is correct and filenames are correct.\n"
		  << e.what()
		  << std::endl;
    }
    catch (const std::exception &e)
    {
	std::cout << "Unhandled exception!\n" << e.what() << std::endl;
	throw e;
    }
}

int main(int argc, char **argv)
{
    std::exception_ptr pExc = nullptr;

    if (argc != 2)
    {
	std::cerr << "One argument is required."
		  << "\nUsage: " << argv[0] << " filename"
		  << "\nfilename = the name of the (.json) file that contains"
		  << " the necessary inputs."
		  << std::endl;
	
	return EXIT_FAILURE;
    }
    
    try
    {
	json data;
	readFileToJSON(argv[1], data);
	
	data["newThing"] = "this really cool new thing";
	data["vector of things"] = {1, 2, 4, 1337, 3.1415};
	
	std::stringstream ss;
	ss << _XSTRINGIFY(DATA_PATH) << "output.json";
	
	writeJSONToFile(ss.str(), data);
    }
    catch (const std::exception &e)
    {
	pExc = std::current_exception();
	handleExceptions(pExc);
	
	return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
