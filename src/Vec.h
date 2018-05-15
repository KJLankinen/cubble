// -*- C++ -*-

#pragma once

#include <math.h>
#include <iostream>
#include <array>
#include <assert.h>

#include "include/json.hpp"

#include "Util.h"

namespace cubble
{   
    // ASSUMPTION: At maximum 4 dimensional vectors.
    template <typename T, size_t SIZE>
    class vec
    {
    public:
	vec()
	{
	    static_assert(SIZE < 5,
			  "Vectors of more than 4 dimension aren't supported.");
	    components.fill((T)0);
	}
	
	/*vec(const vec<T, SIZE> &o) : vec()
	{
	    *this = o;
	}*/
	
	template <typename T2>
	vec(const vec<T2, SIZE> &o)
	    : vec()
	{
	    for (size_t i = 0; i < SIZE; ++i)
		components[i] = (T)o[i];
	}

	vec(std::initializer_list<T> list)
	    : vec()
	{
	    assert(list.size() == SIZE);
	    size_t i = 0;
	    for (const auto &val : list)
	    {
		components[i] = val;
		++i;
	    }
	}
	
	~vec() {}

	T getSquaredLength() const
	{
	    T temp = 0;

	    for (const T &val : components)
		temp += val * val;
	    
	    return temp;
	}
	
	T getLength() const { return std::sqrt(getSquaredLength()); }

	void setComponent(T val, size_t i)
	{
	    assert(i < components.size());
	    components[i] = val;
	}

	static const std::string& getComponentName(size_t i)
	{   
	    const static std::array<std::string, 4> names = {"x", "y", "z", "w"};
	    return names[i];
	}
	
	// + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + 
	friend vec<T, SIZE> operator+(vec<T, SIZE> copy, const vec<T, SIZE> &o)
	{
	    copy += o;
	    return copy;
	}
	
	friend vec<T, SIZE> operator+(vec<T, SIZE> copy, const vec<T, SIZE> &&o)
	{
	    copy += o;
	    return copy;
	}
	
	friend vec<T, SIZE> operator+(vec<T, SIZE> copy, T s)
	{
	    copy += s;
	    return copy;
	}

	
	// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	friend vec<T, SIZE> operator-(vec<T, SIZE> copy, const vec<T, SIZE> &o)
	{
	    copy -= o;
	    return copy;
	}

	friend vec<T, SIZE> operator-(vec<T, SIZE> copy, const vec<T, SIZE> &&o)
	{
	    copy -= o;
	    return copy;
	}

	friend vec<T, SIZE> operator-(vec<T, SIZE> copy, T s)
	{
	    copy -= s;
	    return copy;
	}

	friend vec<T, SIZE> operator-(vec<T, SIZE> copy)
	{
	    copy -= (T)2 * copy;
	    return copy;
	}

	
	// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
	friend vec<T, SIZE> operator*(vec<T, SIZE> copy, const vec<T, SIZE> &o)
	{
	    copy *= o;
	    return copy;
	}
	
	friend vec<T, SIZE> operator*(vec<T, SIZE> copy, const vec<T, SIZE> &&o)
	{
	    copy *= o;
	    return copy;
	}

	friend vec<T, SIZE> operator*(vec<T, SIZE> copy, T s)
	{
	    copy *= s;
	    return copy;
	}
	
	friend vec<T, SIZE> operator*(T s, vec<T, SIZE> copy)
	{
	    copy *= s;
	    return copy;
	}

	
	// / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / /
	friend vec<T, SIZE> operator/(vec<T, SIZE> copy, const vec<T, SIZE> &o)
	{
	    copy /= o;
	    return copy;
	}

	friend vec<T, SIZE> operator/(vec<T, SIZE> copy, const vec<T, SIZE> &&o)
	{
	    copy /= o;
	    return copy;
	}

	friend vec<T, SIZE> operator/(vec<T, SIZE> copy, T s)
	{
	    copy /= s;
	    return copy;
	}

	
	// % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
	friend vec<T, SIZE> operator%(vec<T, SIZE> copy, const vec<T, SIZE> &o)
	{
	    copy %= o;
	    return copy;
	}
	
	friend vec<T, SIZE> operator%(vec<T, SIZE> copy, const vec<T, SIZE> &&o)
	{
	    copy %= o;
	    return copy;
	}

	friend vec<T, SIZE> operator%(vec<T, SIZE> copy, T s)
	{
	    copy %= s;
	    return copy;
	}

	
	// += += += += += += += += += += += += += += += += += += += += += +=
	friend void operator+=(vec<T, SIZE> &t, const vec<T, SIZE> &o)
	{
	    for (size_t i = 0; i < t.components.size(); ++i)
		t.components[i] += o[i];
	}
	
	friend void operator+=(vec<T, SIZE> &t, const vec<T, SIZE> &&o)
	{
	    for (size_t i = 0; i < t.components.size(); ++i)
		t.components[i] += o[i];
	}

	friend void operator+=(vec<T, SIZE> &t, T s)
	{
	    for (size_t i = 0; i < t.components.size(); ++i)
		t.components[i] += s;
	}


	// -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= 
	friend void operator-=(vec<T, SIZE> &t, const vec<T, SIZE> &o)
	{
	    for (size_t i = 0; i < t.components.size(); ++i)
		t.components[i] -= o[i];
	}

	friend void operator-=(vec<T, SIZE> &t, const vec<T, SIZE> &&o)
	{
	    for (size_t i = 0; i < t.components.size(); ++i)
		t.components[i] -= o[i];
	}

	friend void operator-=(vec<T, SIZE> &t, T s)
	{
	    for (size_t i = 0; i < t.components.size(); ++i)
		t.components[i] -= s;
	}


	// *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= 
	friend void operator*=(vec<T, SIZE> &t, const vec<T, SIZE> &o)
	{
	    for (size_t i = 0; i < t.components.size(); ++i)
		t.components[i] *= o[i];
	}

	friend void operator*=(vec<T, SIZE> &t, const vec<T, SIZE> &&o)
	{
	    for (size_t i = 0; i < t.components.size(); ++i)
		t.components[i] *= o[i];
	}

	friend void operator*=(vec<T, SIZE> &t, T s)
	{
	    for (size_t i = 0; i < t.components.size(); ++i)
		t.components[i] *= s;
	}


	// /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= 
	friend void operator/=(vec<T, SIZE> &t, const vec<T, SIZE> &o)
	{
	    for (size_t i = 0; i < t.components.size(); ++i)
		t.components[i] /= o[i];
	}

	friend void operator/=(vec<T, SIZE> &t, const vec<T, SIZE> &&o)
	{
	    for (size_t i = 0; i < t.components.size(); ++i)
		t.components[i] /= o[i];
	}

	friend void operator/=(vec<T, SIZE> &t, T s)
	{
	    for (size_t i = 0; i < t.components.size(); ++i)
		t.components[i] /= s;
	}


	// %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= 
	friend void operator%=(vec<T, SIZE> &t, vec<T, SIZE> &o)
	{
	    for (size_t i = 0; i < t.components.size(); ++i)
		t.components[i] %= o[i];
	}

	friend void operator%=(vec<T, SIZE> &t, vec<T, SIZE> &&o)
	{
	    for (size_t i = 0; i < t.components.size(); ++i)
		t.components[i] %= o[i];
	}

	friend void operator%=(vec<T, SIZE> &t, T s)
	{
	    for (size_t i = 0; i < t.components.size(); ++i)
		t.components[i] %= s;
	}

	
	// = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
	void operator=(vec<T, SIZE> copy)
	{
	    components.swap(copy.components);
	}

	
	// == == == == == == == == == == == == == == == == == == == == == == 
	friend bool operator==(const vec<T, SIZE> &t, const vec<T, SIZE> &o)
	{
	    bool equal = true;
	    for (size_t i = 0; i < t.components.size(); ++i)
		equal &= t[i] - (T)epsilon <= o[i]
		    && t[i] + (T)epsilon >= o[i];
	    
	    return equal;
	}

	
	// != != != != != != != != != != != != != != != != != != != != != != 
	friend bool operator!=(const vec<T, SIZE> &t, const vec<T, SIZE> &o)
	{
	    return !(t == o);
	}

	// [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] [] 
	T operator[](size_t i) const
	{
	    assert(i < components.size());
	    return components[i];
	}
	
	T operator[](int i) const
	{
	    assert(i >= 0);
	    return this->operator[]((size_t)i);
	}
	
	// << << << << << << << << << << << << << << << << << << << << << << 
	friend std::ostream& operator<<(std::ostream &os, const vec<T, SIZE> &v)
	{
	    for (size_t i = 0; i < v.components.size() - 1; ++i)
		os << v[i] << ", ";

	    os << v.components[v.components.size() - 1];
	    
	    return os;
	}

	friend void to_json(nlohmann::json &j, const vec<T, SIZE> &v)
	{
	    for (size_t i = 0; i < v.components.size(); ++i)
		j[getComponentName(i)] = v[i];
	}
        
	friend void from_json(const nlohmann::json &j, vec<T, SIZE> &v)
	{
	    for (size_t i = 0; i < v.components.size(); ++i)
	        v.components[i] = j[getComponentName(i)];
	}
	
    private:
	std::array<T, SIZE> components;
    };
}
