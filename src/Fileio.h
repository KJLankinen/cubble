// -*- C++ -*-
// For emacs to intepret .h files as C++ instead of C.

#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <assert.h>
#include "include/json.hpp"

using json = nlohmann::json;

namespace cubble
{
    namespace fileio
    {
	struct FileWrapper
	{
	    // A filestream wrapper that handles opening and closing files safely.
	private:
	    FileWrapper(const std::string &filename, bool isInput)
		: filename(filename)
	    {
		assert(!filename.empty());
		std::ios_base::openmode mode = isInput ? std::ios::in : std::ios::out;
		file.open(filename, mode);
		
		if (!file.is_open())
		{
		    perror(("Error while opening file " + filename).c_str());
		    throw std::runtime_error("");
		}
	    }
	    
	    ~FileWrapper()
	    {
		if (file.is_open())
		    file.close();
		
		if (file.bad())
		{
		    perror(("Error while reading file " + filename).c_str());
		    throw std::runtime_error("");
		}	
	    }
	    
	    void operator>>(json &j)
	    {
		file >> j;
	    }
	    
	    void operator<<(const json &j)
	    {
		file << std::setw(4) << j;
	    }
	    
	    std::string filename;
	    std::fstream file;
	    
	    // Only friend functions listed here are allowed to use this implementation.
	    friend void readFileToJSON(const std::string&, json&);
	    friend void writeJSONToFile(const std::string&, const json&);
	};
	
	inline void readFileToJSON(const std::string &filename, json &j)
	{
	    FileWrapper file(filename, true);
	    file >> j;
	}
	
	inline void writeJSONToFile(const std::string &filename, const json &j)
	{
	    FileWrapper file(filename, false);
	    file << j;
	}
    }
}
