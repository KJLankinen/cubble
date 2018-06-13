// -*- C++ -*-

#pragma once

#include <cuda_runtime.h>

namespace cubble
{
    class Cell
    {
    public:
	__host__ __device__
	Cell() {}
	__host__ __device__
	~Cell() {}
	
	int offset = 0;
	int size = 0;
    };
};
