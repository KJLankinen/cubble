#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <assert.h>
#include "include/json.hpp"

using json = nlohmann::json;

namespace Cubble
{
    struct FileWrapper
    {
    public:
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
		file << j;
	    }
	
    private:
	std::string filename;
	std::fstream file;
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
