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

// Macro for (arbitrary) class member variable with getter and setter for CPU
#define CUBBLE_PROP(type, var, defVal)					\
private:								\
    type var = defVal;								\
public:									\
    type get##var() const { return var; }						\
    void set##var(const type &val) { var = val; }


// Macro for (arbitrary) const class member variable with getter for CPU
#define CUBBLE_CONST_PROP(type, var, defVal)				\
private:								\
    type var = defVal;								\
public:									\
    type get##var() const { return var; }						


// Macro for (arbitrary) class member variable with getter and setter, for CPU & GPU
#define CUBBLE_HOST_DEVICE_PROP(type, var, defVal)			\
private:								\
    type var = defVal;								\
public:									\
    __host__ __device__ type get##var() const { return var; }			\
    __host__ __device__ void set##var(const type &val) { var = val; }
