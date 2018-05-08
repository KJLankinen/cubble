#pragma once

#include <stdexcept>
#include <iostream>
#include <exception>

#include "include/json.hpp"

namespace cubble
{
    namespace exception
    {
	inline void handleException(const std::exception_ptr pExc)
	{
	    using json = nlohmann::json; 
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
    }

    const double epsilon = 1.0e-10;
}
