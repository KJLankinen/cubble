// -*- C++ -*-
// For emacs to intepret .h files as C++ instead of C.

#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <assert.h>

#ifndef __CUDACC__
  #include "include/json.hpp"
#endif

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
	    
	    void operator>>(nlohmann::json &j)
	    {
		file >> j;
	    }
	    
	    void operator<<(const nlohmann::json &j)
	    {
		file << std::setw(4) << j;
	    }

	    template <typename T>
	    void operator<<(const T &val)
	    {
		file << val << "\n";
	    }
	    
	    std::string filename;
	    std::fstream file;
	    
	    // Only friend functions listed here are allowed to use this implementation.
	    friend void readFileToJSON(const std::string&, nlohmann::json&);
	    friend void writeJSONToFile(const std::string&, const nlohmann::json&);
	    template <typename T>
	    friend void writeVectorToFile(const std::string&, const std::vector<T>&);
	    template <typename T>
	    friend void writeVectorToFile(const std::string&, const std::vector<T*>&);
	};
	
	void readFileToJSON(const std::string &filename, nlohmann::json &j)
	{
	    FileWrapper file(filename, true);
	    file >> j;
	}
	
	void writeJSONToFile(const std::string &filename, const nlohmann::json &j)
	{
	    FileWrapper file(filename, false);
	    file << j;
	}

	template <typename T>
	void writeVectorToFile(const std::string &filename, const std::vector<T> &v)
	{
	    FileWrapper file(filename, false);
	    for (const auto &val : v)
		file << val;
	}

	template <typename T>
	void writeVectorToFile(const std::string &filename, const std::vector<T*> &v)
	{
	    FileWrapper file(filename, false);
	    for (const auto *val : v)
	    {
		assert(val);
		file << *val;
	    }
	}
    }
}
