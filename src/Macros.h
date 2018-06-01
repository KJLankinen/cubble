//-*- C++ -*-
#pragma once

#include <cuda.h>
#include <curand.h>

#define CUBBLE_XSTRINGIFY(s) CUBBLE_STRINGIFY(s)
#define CUBBLE_STRINGIFY(s) #s
#define CUBBLE_IO_PARAMETER(read, j, param)		\
    do							\
    {							\
	if (read)					\
	{						\
	    param = j[#param];				\
	    std::string	s(#param);	\
	    s += "Expl";				\
	    std::cout << j[s] << ": " << param;		\
	    std::cout << std::endl;			\
	}						\
	else						\
	{						\
	    j[#param] = param;				\
	}						\
    }							\
    while(0)

#define CUDA_CALL(x)							\
    do									\
    {									\
	cudaError_t result = x;						\
	if (result != cudaSuccess)					\
	{								\
	    std::cerr << "Error at " << __FILE__ << ":" << __LINE__;	\
	    std::cerr << ": " << cudaGetErrorName(result) << "\n"	\
		      << cudaGetErrorString(result) << std::endl;	\
	}								\
    }									\
    while(0)

#define CURAND_CALL(x)							\
    do									\
    {									\
        curandStatus_t result = x;					\
	if (result != CURAND_STATUS_SUCCESS)				\
	{								\
	    std::cerr << "Error at " << __FILE__ << ":" << __LINE__;	\
	    std::cerr << result << std::endl;				\
	}								\
    }									\
    while(0)

#define CUBBLE_PROP(type, var)						\
    private:								\
    type var;								\
public:								\
type get##var() { return var; }					\
void set##var(type val) { var = val; }
