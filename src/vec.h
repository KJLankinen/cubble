/*
    Cubble
    Copyright (C) 2019  Juhana Lankinen

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "nlohmann/json.hpp"

namespace cubble {
template <typename T> struct vec {
    T x = 0;
    T y = 0;
    T z = 0;

    __host__ __device__ vec() {}

    __host__ __device__ vec(T x) : x(x), y(x), z(x) {}

    template <typename T2> __host__ __device__ vec(const vec<T2> &o) {
        x = static_cast<T>(o.x);
        y = static_cast<T>(o.y);
        z = static_cast<T>(o.z);
    }

    __host__ __device__ vec(T x, T y, T z) : x(x), y(y), z(z) {}

    __host__ __device__ ~vec() {}

    // + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
    __host__ __device__ friend vec<T> operator+(vec<T> copy, const vec<T> &o) {
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

    // << << << << << << << << << << << << << << << << << << << << << <<
    friend std::ostream &operator<<(std::ostream &os, const vec<T> &v) {
        os << v.x << ", " << v.y << ", " << v.z;

        return os;
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
};

template <typename T> __host__ __device__ T lengthSq(const vec<T> &lhs) {
    T temp = 0;

    temp += lhs.x * lhs.x;
    temp += lhs.y * lhs.y;
    temp += lhs.z * lhs.z;

    return temp;
}

template <typename T> __host__ __device__ T length(const vec<T> &lhs) {
    return sqrt(lengthSq(lhs));
}

template <typename T> __host__ __device__ vec<T> abs(const vec<T> &lhs) {
    vec<T> v;
    v.x = lhs.x < 0 ? -lhs.x : lhs.x;
    v.y = lhs.y < 0 ? -lhs.y : lhs.y;
    v.z = lhs.z < 0 ? -lhs.z : lhs.z;

    return v;
}

template <typename T> __host__ __device__ vec<T> normalize(const vec<T> &v) {
    return v / length(v);
}

template <typename T> __host__ __device__ T maxComponent(const vec<T> &lhs) {
    return lhs.x > lhs.y ? (lhs.x > lhs.z ? lhs.x : lhs.z)
                         : (lhs.y > lhs.z ? lhs.y : lhs.z);
}

template <typename T> __host__ __device__ T minComponent(const vec<T> &lhs) {
    return lhs.x < lhs.y ? (lhs.x < lhs.z ? lhs.x : lhs.z)
                         : (lhs.y < lhs.z ? lhs.y : lhs.z);
}

template <typename S, typename T> struct SameType {
    static constexpr bool value = false;
};

template <typename T> struct SameType<T, T> {
    static constexpr bool value = true;
};

template <typename T> __host__ __device__ vec<T> ceil(const vec<T> &lhs) {
    if constexpr (SameType<T, float>::value) {
        return vec<T>(ceilf(lhs.x), ceilf(lhs.y), ceilf(lhs.z));
    } else if constexpr (SameType<T, double>::value) {
        return vec<T>(::ceil(lhs.x), ::ceil(lhs.y), ::ceil(lhs.z));
    } else {
        return lhs;
    }
}

template <typename T> __host__ __device__ vec<T> floor(const vec<T> &lhs) {
    if constexpr (SameType<T, float>::value) {
        return vec<T>(floorf(lhs.x), floorf(lhs.y), floorf(lhs.z));
    } else if constexpr (SameType<T, double>::value) {
        return vec<T>(::floor(lhs.x), ::floor(lhs.y), ::floor(lhs.z));
    } else {
        return lhs;
    }
}

template <typename T>
__host__ __device__ bool operator==(const vec<T> &lhs, const vec<T> &rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

template <typename T>
__host__ __device__ bool operator!=(const vec<T> &lhs, const vec<T> &rhs) {
    return !(lhs == rhs);
}

typedef vec<float> fvec;
typedef vec<double> dvec;
typedef vec<int32_t> ivec;
typedef vec<uint32_t> uvec;
} // namespace cubble
