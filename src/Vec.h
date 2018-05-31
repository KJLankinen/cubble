// -*- C++ -*-

#pragma once

#include <math.h>
#include <iostream>
#include <assert.h>

#include "include/json.hpp"

#include "Util.h"

namespace cubble
{
    template <typename T>
    class vec
    {
    public:
	vec()
	{}
	
	template <typename T2>
	vec(const vec<T2> &o)
	{
	    x = (T)o.x;
	    y = (T)o.y;
#if (NUM_DIM == 3)
	    z = (T)o.z;
#endif
	}
	
#if (NUM_DIM == 3)
	vec(T x, T y, T z)
	    : x(x)
	    , y(y)
	    , z(z)
	{}
#else
	vec(T x, T y)
	    : x(x)
	    , y(y)
	{}
#endif
	
	~vec() {}

	T getSquaredLength() const
	{
	    T temp = 0;

	    temp += x * x;
	    temp += y * y;
#if (NUM_DIM == 3)
	    temp += z * z;
#endif
	    
	    return temp;
	}
	
	T getLength() const { return std::sqrt(getSquaredLength()); }

	vec<T> getAbsolute() const
	{
	    vec<T> v;
	    v.x = std::abs(x);
	    v.y = std::abs(y);
#if (NUM_DIM == 3)
	    v.z = std::abs(z);
#endif

	    return v;
	}

	T getMaxComponent() const
	{
#if (NUM_DIM == 3)
	    return x > y ? (x > z ? x : z) : (y > z ? y : z);
#else
	    return x > y ? x : y;
#endif
	}
	
	T getMinComponent() const
	{
#if (NUM_DIM == 3)
	    return x < y ? (x < z ? x : z) : (y < z ? y : z);
#else
	    return x < y ? x : y;
#endif
	}
	
	template <typename T2>
	vec<T2> asType() const
	{
	    vec<T2> v(*this);

	    return v;
	}
	
	// + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
	friend vec<T> operator+(vec<T> copy, const vec<T> &o)
	{
	    copy += o;
	    return copy;
	}
	
	friend vec<T> operator+(vec<T> copy, const vec<T> &&o)
	{
	    copy += o;
	    return copy;
	}
	
	friend vec<T> operator+(vec<T> copy, T s)
	{
	    copy += s;
	    return copy;
	}

	
	// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	friend vec<T> operator-(vec<T> copy, const vec<T> &o)
	{
	    copy -= o;
	    return copy;
	}

	friend vec<T> operator-(vec<T> copy, const vec<T> &&o)
	{
	    copy -= o;
	    return copy;
	}

	friend vec<T> operator-(vec<T> copy, T s)
	{
	    copy -= s;
	    return copy;
	}

	friend vec<T> operator-(vec<T> copy)
	{
	    copy -= (T)2 * copy;
	    return copy;
	}

	
	// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	friend vec<T> operator*(vec<T> copy, const vec<T> &o)
	{
	    copy *= o;
	    return copy;
	}
	
	friend vec<T> operator*(vec<T> copy, const vec<T> &&o)
	{
	    copy *= o;
	    return copy;
	}

	friend vec<T> operator*(vec<T> copy, T s)
	{
	    copy *= s;
	    return copy;
	}
	
	friend vec<T> operator*(T s, vec<T> copy)
	{
	    copy *= s;
	    return copy;
	}

	
	// / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / /
	friend vec<T> operator/(vec<T> copy, const vec<T> &o)
	{
	    copy /= o;
	    return copy;
	}

	friend vec<T> operator/(vec<T> copy, const vec<T> &&o)
	{
	    copy /= o;
	    return copy;
	}

	friend vec<T> operator/(vec<T> copy, T s)
	{
	    copy /= s;
	    return copy;
	}

	
	// % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
	friend vec<T> operator%(vec<T> copy, const vec<T> &o)
	{
	    copy %= o;
	    return copy;
	}
	
	friend vec<T> operator%(vec<T> copy, const vec<T> &&o)
	{
	    copy %= o;
	    return copy;
	}

	friend vec<T> operator%(vec<T> copy, T s)
	{
	    copy %= s;
	    return copy;
	}

	
	// += += += += += += += += += += += += += += += += += += += += += +=
	friend void operator+=(vec<T> &t, const vec<T> &o)
	{
	    t.x += o.x;
	    t.y += o.y;
#if (NUM_DIM == 3)
	    t.z += o.z;
#endif
	}
	
	friend void operator+=(vec<T> &t, const vec<T> &&o)
	{
	    t.x += o.x;
	    t.y += o.y;
#if (NUM_DIM == 3)
	    t.z += o.z;
#endif
	}

	friend void operator+=(vec<T> &t, T s)
	{
	    t.x += s;
	    t.y += s;
#if (NUM_DIM == 3)
	    t.z += s;
#endif
	}


	// -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= 
	friend void operator-=(vec<T> &t, const vec<T> &o)
	{
	    t.x -= o.x;
	    t.y -= o.y;
#if (NUM_DIM == 3)
	    t.z -= o.z;
#endif
	}

	friend void operator-=(vec<T> &t, const vec<T> &&o)
	{
	    t.x -= o.x;
	    t.y -= o.y;
#if (NUM_DIM == 3)
	    t.z -= o.z;
#endif
	}

	friend void operator-=(vec<T> &t, T s)
	{
	    t.x -= s;
	    t.y -= s;
#if (NUM_DIM == 3)
	    t.z -= s;
#endif
	}


	// *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= 
	friend void operator*=(vec<T> &t, const vec<T> &o)
	{
	    t.x *= o.x;
	    t.y *= o.y;
#if (NUM_DIM == 3)
	    t.z *= o.z;
#endif
	}

	friend void operator*=(vec<T> &t, const vec<T> &&o)
	{
	    t.x *= o.x;
	    t.y *= o.y;
#if (NUM_DIM == 3)
	    t.z *= o.z;
#endif
	}

	friend void operator*=(vec<T> &t, T s)
	{
	    t.x *= s;
	    t.y *= s;
#if (NUM_DIM == 3)
	    t.z *= s;
#endif
	}


	// /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= 
	friend void operator/=(vec<T> &t, const vec<T> &o)
	{
	    t.x /= o.x;
	    t.y /= o.y;
#if (NUM_DIM == 3)
	    t.z /= o.z;
#endif
	}

	friend void operator/=(vec<T> &t, const vec<T> &&o)
	{
	    t.x /= o.x;
	    t.y /= o.y;
#if (NUM_DIM == 3)
	    t.z /= o.z;
#endif
	}

	friend void operator/=(vec<T> &t, T s)
	{
	    t.x /= s;
	    t.y /= s;
#if (NUM_DIM == 3)
	    t.z /= s;
#endif
	}


	// %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= 
	friend void operator%=(vec<T> &t, vec<T> &o)
	{
	    t.x %= o.x;
	    t.y %= o.y;
#if (NUM_DIM == 3)
	    t.z %= o.z;
#endif
	}

	friend void operator%=(vec<T> &t, vec<T> &&o)
	{
	    t.x %= o.x;
	    t.y %= o.y;
#if (NUM_DIM == 3)
	    t.z %= o.z;
#endif
	}

	friend void operator%=(vec<T> &t, T s)
	{
	    t.x %= s;
	    t.y %= s;
#if (NUM_DIM == 3)
	    t.z %= s;
#endif
	}

	
	// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
	void operator=(vec<T> copy)
	{
	    x = copy.x;
	    y = copy.y;
#if (NUM_DIM == 3)
	    z = copy.z;
#endif
	}

	
	// == == == == == == == == == == == == == == == == == == == == == == 
	friend bool operator==(const vec<T> &t, const vec<T> &o)
	{
	    bool equal = true;
	    equal &= t.x - (T)epsilon <= o.x && t.x + (T)epsilon >= o.x;
	    equal &= t.y - (T)epsilon <= o.y && t.y + (T)epsilon >= o.y;
#if (NUM_DIM == 3)
	    equal &= t.z - (T)epsilon <= o.z && t.z + (T)epsilon >= o.z;
#endif
	    return equal;
	}

	
	// != != != != != != != != != != != != != != != != != != != != != != 
	friend bool operator!=(const vec<T> &t, const vec<T> &o)
	{
	    return !(t == o);
	}

	
	// << << << << << << << << << << << << << << << << << << << << << << 
	friend std::ostream& operator<<(std::ostream &os, const vec<T> &v)
	{
	    os << v.x << ", " << v.y;
#if (NUM_DIM == 3)
	    os << ", " << v.z;
#endif
	    return os;
	}

	
	// min min min min min min min min min min min min min min min min min min
	friend vec<T> min(const vec<T> &v1, const vec<T> &v2)
	{
	    vec<T> retVec;
	    retVec.x = v1.x < v2.x ? v1.x : v2.x;
	    retVec.y = v1.y < v2.y ? v1.y : v2.y;
#if (NUM_DIM == 3)
	    retVec.z = v1.z < v2.z ? v1.z : v2.z;
#endif
	    
	    return retVec;
	}

	
	// max max max max max max max max max max max max max max max max max max
	friend vec<T> max(const vec<T> &v1, const vec<T> &v2)
	{
	    vec<T> retVec;
	    retVec.x = v1.x > v2.x ? v1.x : v2.x;
	    retVec.y = v1.y > v2.y ? v1.y : v2.y;
#if (NUM_DIM == 3)
	    retVec.z = v1.z > v2.z ? v1.z : v2.z;
#endif
	    
	    return retVec;
	}
	
	
	// .json .json .json .json .json .json .json .json .json .json .json
	friend void to_json(nlohmann::json &j, const vec<T> &v)
	{
	    j["x"] = v.x;
	    j["y"] = v.y;
#if (NUM_DIM == 3)
	    j["z"] = v.z;
#endif
	}
        
	friend void from_json(const nlohmann::json &j, vec<T> &v)
	{
	    v.x = j["x"];
	    v.y = j["y"];
#if (NUM_DIM == 3)
	    v.z = j["z"];
#endif
	}
        
        T x = 0;
	T y = 0;
#if (NUM_DIM == 3)
	T z = 0;
#endif
    };

    typedef vec<double> dvec;
    typedef vec<int> ivec;
    typedef vec<size_t> uvec;
}
