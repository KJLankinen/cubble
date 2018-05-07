#include "Bubble.h"

#include <assert.h>

using namespace cubble;

bool Bubble::overlapsWith(const Bubble &b) const
{
    double rads = b.radius + radius;
    return (b.position - position).getSquaredLength() < rads * rads;
}

double Bubble::getOverlapRadiusSquared(const Bubble &b) const
{
    assert(position != b.position);
    assert(radius > 0);
    assert(b.radius > 0);
    
    double d2 = (position - b.position).getSquaredLength();
    double r12 = radius * radius;
    double r22 = b.radius * b.radius;
    double temp = d2 - r12 - r22;
    temp = (r12 * r22 - 0.25 * temp * temp);
    
    return temp > 0 ? temp / d2 : 0;
}
