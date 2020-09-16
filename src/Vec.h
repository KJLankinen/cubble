#pragma once

#include "nlohmann/json.hpp"
#include <assert.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>

#include "Util.h"

namespace cubble {
template <typename T> class vec {
  public:
    __host__ __device__ vec() {}

    __host__ __device__ vec(T x) : x(x), y(x), z(x) {}

    template <typename T2> __host__ __device__ vec(const vec<T2> &o) {
        x = (T)o.x;
        y = (T)o.y;
        z = (T)o.z;
    }

    __host__ __device__ vec(T x, T y, T z) : x(x), y(y), z(z) {}

    __host__ __device__ ~vec() {}

    __host__ __device__ T getSquaredLength() const {
        T temp = 0;

        temp += x * x;
        temp += y * y;
        temp += z * z;

        return temp;
    }

    __host__ __device__ T getLength() const {
        return std::sqrt(getSquaredLength());
    }

    __host__ __device__ vec<T> getAbsolute() const {
        vec<T> v;
        v.x = x < 0 ? -x : x;
        v.y = y < 0 ? -y : y;
        v.z = z < 0 ? -z : z;

        return v;
    }

    __host__ __device__ static vec<T> normalize(vec<T> &v) {
        return v / v.getLength();
    }

    __host__ __device__ static vec<T> normalize(const vec<T> &v) {
        return v / v.getLength();
    }

    __host__ __device__ T getMaxComponent() const {
        return x > y ? (x > z ? x : z) : (y > z ? y : z);
    }

    __host__ __device__ T getMinComponent() const {
        return x < y ? (x < z ? x : z) : (y < z ? y : z);
    }

    template <typename T2> __host__ __device__ vec<T2> asType() const {
        vec<T2> v(*this);

        return v;
    }

    __host__ vec<int> ceil() const {
        return vec<int>(std::ceil(x), std::ceil(y), std::ceil(z));
    }

    __host__ vec<int> floor() const {
        return vec<int>(std::floor(x), std::floor(y), std::floor(z));
    }

    // + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
    __host__ __device__ friend vec<T> operator+(vec<T> copy, const vec<T> &o) {
        copy += o;
        return copy;
    }

    __host__ __device__ friend vec<T> operator+(vec<T> copy, const vec<T> &&o) {
        copy += o;
        return copy;
    }

    __host__ __device__ friend vec<T> operator+(vec<T> copy, T s) {
        copy += s;
        return copy;
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    __host__ __device__ friend vec<T> operator-(vec<T> copy, const vec<T> &o) {
        copy -= o;
        return copy;
    }

    __host__ __device__ friend vec<T> operator-(vec<T> copy, const vec<T> &&o) {
        copy -= o;
        return copy;
    }

    __host__ __device__ friend vec<T> operator-(vec<T> copy, T s) {
        copy -= s;
        return copy;
    }

    __host__ __device__ friend vec<T> operator-(vec<T> copy) {
        copy -= (T)2 * copy;
        return copy;
    }

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    __host__ __device__ friend vec<T> operator*(vec<T> copy, const vec<T> &o) {
        copy *= o;
        return copy;
    }

    __host__ __device__ friend vec<T> operator*(vec<T> copy, const vec<T> &&o) {
        copy *= o;
        return copy;
    }

    __host__ __device__ friend vec<T> operator*(vec<T> copy, T s) {
        copy *= s;
        return copy;
    }

    __host__ __device__ friend vec<T> operator*(T s, vec<T> copy) {
        copy *= s;
        return copy;
    }

    // / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / /
    __host__ __device__ friend vec<T> operator/(vec<T> copy, const vec<T> &o) {
        copy /= o;
        return copy;
    }

    __host__ __device__ friend vec<T> operator/(vec<T> copy, const vec<T> &&o) {
        copy /= o;
        return copy;
    }

    __host__ __device__ friend vec<T> operator/(vec<T> copy, T s) {
        copy /= s;
        return copy;
    }

    __host__ __device__ friend vec<T> operator/(T s, vec<T> copy) {
        copy = vec<T>(s / copy.x, s / copy.y, s / copy.z);
        return copy;
    }

    // % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    __host__ __device__ friend vec<T> operator%(vec<T> copy, const vec<T> &o) {
        copy %= o;
        return copy;
    }

    __host__ __device__ friend vec<T> operator%(vec<T> copy, const vec<T> &&o) {
        copy %= o;
        return copy;
    }

    __host__ __device__ friend vec<T> operator%(vec<T> copy, T s) {
        copy %= s;
        return copy;
    }

    // += += += += += += += += += += += += += += += += += += += += += +=
    __host__ __device__ friend void operator+=(vec<T> &t, const vec<T> &o) {
        t.x += o.x;
        t.y += o.y;
        t.z += o.z;
    }

    __host__ __device__ friend void operator+=(vec<T> &t, const vec<T> &&o) {
        t.x += o.x;
        t.y += o.y;
        t.z += o.z;
    }

    __host__ __device__ friend void operator+=(vec<T> &t, T s) {
        t.x += s;
        t.y += s;
        t.z += s;
    }

    // -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -=
    __host__ __device__ friend void operator-=(vec<T> &t, const vec<T> &o) {
        t.x -= o.x;
        t.y -= o.y;
        t.z -= o.z;
    }

    __host__ __device__ friend void operator-=(vec<T> &t, const vec<T> &&o) {
        t.x -= o.x;
        t.y -= o.y;
        t.z -= o.z;
    }

    __host__ __device__ friend void operator-=(vec<T> &t, T s) {
        t.x -= s;
        t.y -= s;
        t.z -= s;
    }

    // *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *=
    __host__ __device__ friend void operator*=(vec<T> &t, const vec<T> &o) {
        t.x *= o.x;
        t.y *= o.y;
        t.z *= o.z;
    }

    __host__ __device__ friend void operator*=(vec<T> &t, const vec<T> &&o) {
        t.x *= o.x;
        t.y *= o.y;
        t.z *= o.z;
    }

    __host__ __device__ friend void operator*=(vec<T> &t, T s) {
        t.x *= s;
        t.y *= s;
        t.z *= s;
    }

    // /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /=
    __host__ __device__ friend void operator/=(vec<T> &t, const vec<T> &o) {
        t.x /= o.x;
        t.y /= o.y;
        t.z /= o.z;
    }

    __host__ __device__ friend void operator/=(vec<T> &t, const vec<T> &&o) {
        t.x /= o.x;
        t.y /= o.y;
        t.z /= o.z;
    }

    __host__ __device__ friend void operator/=(vec<T> &t, T s) {
        t.x /= s;
        t.y /= s;
        t.z /= s;
    }

    // %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %=
    __host__ __device__ friend void operator%=(vec<T> &t, vec<T> &o) {
        t.x %= o.x;
        t.y %= o.y;
        t.z %= o.z;
    }

    __host__ __device__ friend void operator%=(vec<T> &t, vec<T> &&o) {
        t.x %= o.x;
        t.y %= o.y;
        t.z %= o.z;
    }

    __host__ __device__ friend void operator%=(vec<T> &t, T s) {
        t.x %= s;
        t.y %= s;
        t.z %= s;
    }

    // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    __host__ __device__ void operator=(vec<T> copy) {
        x = copy.x;
        y = copy.y;
        z = copy.z;
    }

    // == == == == == == == == == == == == == == == == == == == == == ==
    __host__ __device__ friend bool operator==(const vec<T> &t,
                                               const vec<T> &o) {
        const T epsilon = (T)(1.0 / 1e6);
        bool equal = true;
        equal &= t.x - epsilon <= o.x && t.x + epsilon >= o.x;
        equal &= t.y - epsilon <= o.y && t.y + epsilon >= o.y;
        equal &= t.z - epsilon <= o.z && t.z + epsilon >= o.z;

        return equal;
    }

    // != != != != != != != != != != != != != != != != != != != != != !=
    __host__ __device__ friend bool operator!=(const vec<T> &t,
                                               const vec<T> &o) {
        return !(t == o);
    }

    // << << << << << << << << << << << << << << << << << << << << << <<
    friend std::ostream &operator<<(std::ostream &os, const vec<T> &v) {
        os << v.x << ", " << v.y << ", " << v.z;

        return os;
    }

    // min min min min min min min min min min min min min min min min min min
    __host__ __device__ friend vec<T> min(const vec<T> &v1, const vec<T> &v2) {
        vec<T> retVec;
        retVec.x = v1.x < v2.x ? v1.x : v2.x;
        retVec.y = v1.y < v2.y ? v1.y : v2.y;
        retVec.z = v1.z < v2.z ? v1.z : v2.z;

        return retVec;
    }

    // max max max max max max max max max max max max max max max max max max
    __host__ __device__ friend vec<T> max(const vec<T> &v1, const vec<T> &v2) {
        vec<T> retVec;
        retVec.x = v1.x > v2.x ? v1.x : v2.x;
        retVec.y = v1.y > v2.y ? v1.y : v2.y;
        retVec.z = v1.z > v2.z ? v1.z : v2.z;

        return retVec;
    }

    // .json .json .json .json .json .json .json .json .json .json .json
    friend void to_json(nlohmann::json &j, const vec<T> &v) {
        j["x"] = v.x;
        j["y"] = v.y;
        j["z"] = v.z;
    }

    friend void from_json(const nlohmann::json &j, vec<T> &v) {
        v.x = j["x"];
        v.y = j["y"];
        v.z = j["z"];
    }

    T x = 0;
    T y = 0;
    T z = 0;
};

typedef vec<float> fvec;
typedef vec<double> dvec;
typedef vec<int> ivec;
typedef vec<uint32_t> uvec;
} // namespace cubble
