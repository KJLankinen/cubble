// -*- C++ -*-
#pragma once

#include <stdexcept>
#include <iostream>
#include <exception>

#ifndef __CUDACC__
  #include "include/json.hpp"
#endif

namespace cubble
{
    inline void handleException(const std::exception_ptr pExc)
    {
#ifndef __CUDACC__
	using json = nlohmann::json;
#endif
	try
	{
	    if (pExc)
		std::rethrow_exception(pExc);
	}
#ifndef __CUDACC__
	catch (const json::exception &e)
	{
	    std::cout << "Encountered a json parse error."
		      << "\nMake sure the .json file is correct and filenames are correct.\n"
		      << e.what()
		      << std::endl;
	}
#endif
	catch (const std::exception &e)
	{
	    std::cout << "Unhandled exception!\n" << e.what() << std::endl;
	    throw e;
	}
    }
    
    const double epsilon = 1.0e-10;
}
