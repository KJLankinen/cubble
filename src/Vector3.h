// -*- C++ -*-

#pragma once

#include <math.h>
#include <iostream>

#include "include/json.hpp"

#include "Util.h"

namespace cubble
{
    template <typename T>
    class Vector3
    {
    public:
	Vector3<T>(T x, T y, T z)
	: x(x)
	    , y(y)
	    , z(z)
	{}
	Vector3<T>() {}
	Vector3<T>(const Vector3<T> &o) { *this = o; }
	~Vector3<T>() {}
	T getSquaredLength() const { return x * x + y * y + z * z; }
	T getLength() const { return std::sqrt(getSquaredLength()); }
	void setX(T newX) { x = newX;}
	void setY(T newY) { y = newY;}
	void setZ(T newZ) { z = newZ;}
	T getX() const { return x; }
	T getY() const { return y; }
	T getZ() const { return z; }
	T dot(const Vector3<T> &o) const { return o.x * x + o.y * y + o.z * z; }
	
	Vector3<T> cross(const Vector3<T> &o) const
	{
	    Vector3<T> v;
	    v.x = y * o.z - z * o.y;
	    v.y = z * o.x - x * o.z;
	    v.z = x * o.y - y * o.x;
	    
	    return v;
	}
	
	Vector3<T> operator+(const Vector3<T> &o) const
	{
	    return Vector3<T>(x + o.x, y + o.y, z + o.z);
	}
	
	Vector3<T> operator-(const Vector3<T> &o) const
	{
	    return Vector3<T>(x - o.x, y - o.y, z - o.z);
	}

	Vector3<T> operator-() const
	{
	    return Vector3<T>(-x, -y, -z);
	}
	
	Vector3<T> operator*(T s) const
	{
	    return Vector3<T>(s * x, s * y, s * z);
	}

	template <typename T2>
	Vector3<T2> operator*(T2 s) const
	{
	    return Vector3<T2>((T2)(s * x), (T2)(s * y), (T2)(s * z));
	}

	Vector3<T> operator*(const Vector3<T> &o) const
	{
	    return Vector3<T>(x * o.x, y * o.y, z * o.z);
	}

	Vector3<T> operator/(const Vector3<T> &o) const
	{
	    return Vector3<T>(x / o.x, y / o.y, z / o.z);
	}

	Vector3<T> operator%(const Vector3<T> &o) const
	{
	    return Vector3<T>(x % o.x, y % o.y, z % o.z);
	}

	void operator+=(const Vector3<T> &o)
	{
	    *this = *this + o;
	}
	
	void operator-=(const Vector3<T> &o)
	{
	    *this = *this - o;
	}
	
	void operator*=(const Vector3<T> &o)
	{
	    *this = *this * o;
	}

	void operator*=(T s)
	{
	    *this = *this * s;
	}
	
	void operator=(const Vector3<T> &o)
	{
	    if (*this == o)
		return;
	    
	    x = o.x;
	    y = o.y;
	    z = o.z;
	}

	template <typename T2>
	void operator=(const Vector3<T2> &o)
	{
	    x = (T)o.getX();
	    y = (T)o.getY();
	    z = (T)o.getZ();
	}
	
	bool operator==(const Vector3<T> &o) const
	{
	    bool equal = true;
	    equal &= (x - (T)epsilon < o.x && x + (T)epsilon > o.x);
	    equal &= (y - (T)epsilon < o.y && y + (T)epsilon > o.y);
	    equal &= (z - (T)epsilon < o.z && z + (T)epsilon > o.z);
	    
	    return equal;
	}

	bool operator!=(const Vector3<T> &o) const
	{
	    return !(*this == o);
	}
	
	friend std::ostream& operator<<(std::ostream &os, const Vector3<T> &v)
	{
	    os << v.x << ", " << v.y << ", " << v.z;
	    
	    return os;
	}
	
    private:
	T x = 0;
	T y = 0;
	T z = 0;
    };

    template <typename T>
    void to_json(nlohmann::json &j, const Vector3<T> &v)
    {
	j = nlohmann::json{
	    {"x", v.getX()},
	    {"y", v.getY()},
	    {"z", v.getZ()}};
    }

    template <typename T>
    void from_json(const nlohmann::json &j, Vector3<T> &v)
    {
	v.setX(j.at("x").get<T>());
	v.setY(j.at("y").get<T>());
	v.setZ(j.at("z").get<T>());
    }

    template <typename T>
    Vector3<T> operator*(T s, const Vector3<T> &o)
    {
	return o * s;
    }
}
