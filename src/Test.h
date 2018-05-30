// -*- C++ -*-

#pragma once

namespace cubble
{
    class Test
    {
    public:
	double x = 0;
	double y = 0;
	double z = 0;
	
	Test() {}
	~Test() {}
	
	void testFunctionWrapper();
    };

    // Kernel has to be defined outside class
    __global__
    void testFunction(float *a, float *b);
};
