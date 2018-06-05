//-*- C++ -*-
#pragma once

#define CUBBLE_XSTRINGIFY(s) CUBBLE_STRINGIFY(s)
#define CUBBLE_STRINGIFY(s) #s

// Macro for reading and writing parameters from/to .json file
#define CUBBLE_IO_PARAMETER(read, j, param)		\
    do							\
    {							\
	if (read)					\
	{						\
	    param = j[#param];				\
	    std::string	s(#param);			\
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

// Cuda error checking
// See Util.h for actual implementation
#ifndef NDEBUG
  #define CUDA_CALL(result) {cubble::cudaAssert((result), __FILE__, __LINE__);}
#else
  #define CUDA_CALL(result) {(result);}
#endif

// Curand error checking
// See Util.h for actual implementation
#ifndef NDEBUG
  #define CURAND_CALL(result) {cubble::curandAssert((result), __FILE__, __LINE__);}
#else
  #define CURAND_CALL(result) {(result);}
#endif

// Macro for (arbitrary) class member variable with getter and setter
#define CUBBLE_PROP(type, var)						\
private:								\
    type var;								\
public:									\
    type get##var() { return var; }						\
    void set##var(type val) { var = val; }
