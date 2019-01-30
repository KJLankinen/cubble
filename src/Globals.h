#pragma once

#if (CUBBLE_FLOAT_TYPE == CUBBLE_FLOAT)
typedef float CubbleFloatType;
#else // default to double precision
typedef double CubbleFloatType;
#endif

const CubbleFloatType CUBBLE_EPSILON = 1.0e-10;
#if NUM_DIM == 3
const int CUBBLE_NUM_NEIGHBORS = 13;
#else
const int CUBBLE_NUM_NEIGHBORS = 4;
#endif