// -*- C++ -*-

#pragma once

#include <cuda_runtime.h>

namespace cubble
{
    class Cell
    {
    public:
	Cell() {}
	~Cell() {}
	
	int offset = 0;
	int size = 0;
    };
};
