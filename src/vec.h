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

#include "macros.h"
#include "nlohmann/json.hpp"

namespace cubble {
template <typename T> struct vec {
    T x = 0;
    T y = 0;
    T z = 0;

    HOST DEVICE vec() {}

    HOST DEVICE vec(T t) : x(t), y(t), z(t) {}

    template <typename T2> HOST DEVICE vec(const vec<T2> &o) {
        x = static_cast<T>(o.x);
        y = static_cast<T>(o.y);
        z = static_cast<T>(o.z);
    }

    HOST DEVICE vec(T t1, T t2, T t3) : x(t1), y(t2), z(t3) {}

    HOST DEVICE ~vec() {}

    // + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + + +
    HOST DEVICE friend vec<T> operator+(vec<T> copy, const vec<T> &o) {
        copy += o;
        return copy;
    }

    HOST DEVICE friend vec<T> operator+(vec<T> copy, T s) {
        copy += s;
        return copy;
    }

    // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    HOST DEVICE friend vec<T> operator-(vec<T> copy, const vec<T> &o) {
        copy -= o;
        return copy;
    }

    HOST DEVICE friend vec<T> operator-(vec<T> copy, T s) {
        copy -= s;
        return copy;
    }

    HOST DEVICE friend vec<T> operator-(vec<T> copy) {
        copy -= (T)2 * copy;
        return copy;
    }

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
    HOST DEVICE friend vec<T> operator*(vec<T> copy, const vec<T> &o) {
        copy *= o;
        return copy;
    }

    HOST DEVICE friend vec<T> operator*(vec<T> copy, T s) {
        copy *= s;
        return copy;
    }

    HOST DEVICE friend vec<T> operator*(T s, vec<T> copy) {
        copy *= s;
        return copy;
    }

    // / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / / /
    HOST DEVICE friend vec<T> operator/(vec<T> copy, const vec<T> &o) {
        copy /= o;
        return copy;
    }

    HOST DEVICE friend vec<T> operator/(vec<T> copy, T s) {
        copy /= s;
        return copy;
    }

    HOST DEVICE friend vec<T> operator/(T s, vec<T> copy) {
        copy = vec<T>(s / copy.x, s / copy.y, s / copy.z);
        return copy;
    }

    // % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    HOST DEVICE friend vec<T> operator%(vec<T> copy, const vec<T> &o) {
        copy %= o;
        return copy;
    }

    HOST DEVICE friend vec<T> operator%(vec<T> copy, T s) {
        copy %= s;
        return copy;
    }

    // += += += += += += += += += += += += += += += += += += += += += +=
    HOST DEVICE friend void operator+=(vec<T> &t, const vec<T> &o) {
        t.x += o.x;
        t.y += o.y;
        t.z += o.z;
    }

    HOST DEVICE friend void operator+=(vec<T> &t, T s) {
        t.x += s;
        t.y += s;
        t.z += s;
    }

    // -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -=
    HOST DEVICE friend void operator-=(vec<T> &t, const vec<T> &o) {
        t.x -= o.x;
        t.y -= o.y;
        t.z -= o.z;
    }

    HOST DEVICE friend void operator-=(vec<T> &t, T s) {
        t.x -= s;
        t.y -= s;
        t.z -= s;
    }

    // *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *= *=
    HOST DEVICE friend void operator*=(vec<T> &t, const vec<T> &o) {
        t.x *= o.x;
        t.y *= o.y;
        t.z *= o.z;
    }

    HOST DEVICE friend void operator*=(vec<T> &t, T s) {
        t.x *= s;
        t.y *= s;
        t.z *= s;
    }

    // /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /= /=
    HOST DEVICE friend void operator/=(vec<T> &t, const vec<T> &o) {
        t.x /= o.x;
        t.y /= o.y;
        t.z /= o.z;
    }

    HOST DEVICE friend void operator/=(vec<T> &t, T s) {
        t.x /= s;
        t.y /= s;
        t.z /= s;
    }

    // %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %= %=
    HOST DEVICE friend void operator%=(vec<T> &t, vec<T> &o) {
        t.x %= o.x;
        t.y %= o.y;
        t.z %= o.z;
    }

    HOST DEVICE friend void operator%=(vec<T> &t, T s) {
        t.x %= s;
        t.y %= s;
        t.z %= s;
    }

    // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    HOST DEVICE void operator=(vec<T> copy) {
        x = copy.x;
        y = copy.y;
        z = copy.z;
    }

    // << << << << << << << << << << << << << << << << << << << << << <<
    friend std::ostream &operator<<(std::ostream &os, const vec<T> &v) {
        os << v.x << ", " << v.y << ", " << v.z;

        return os;
    }
};

template <typename T> void to_json(nlohmann::json &j, const vec<T> &v) {
    j = nlohmann::json{
        {"x", v.x},
        {"y", v.y},
        {"z", v.z},
    };
}

template <typename T> void from_json(const nlohmann::json &j, vec<T> &v) {
    j.at("x").get_to(v.x);
    j.at("y").get_to(v.y);
    j.at("z").get_to(v.z);
}

template <typename T> HOST DEVICE T lengthSq(const vec<T> &lhs) {
    T temp = 0;

    temp += lhs.x * lhs.x;
    temp += lhs.y * lhs.y;
    temp += lhs.z * lhs.z;

    return temp;
}

template <typename T> HOST DEVICE T length(const vec<T> &lhs) {
    return sqrt(lengthSq(lhs));
}

template <typename T> HOST DEVICE vec<T> abs(const vec<T> &lhs) {
    vec<T> v;
    v.x = lhs.x < 0 ? -lhs.x : lhs.x;
    v.y = lhs.y < 0 ? -lhs.y : lhs.y;
    v.z = lhs.z < 0 ? -lhs.z : lhs.z;

    return v;
}

template <typename T> HOST DEVICE vec<T> normalize(const vec<T> &v) {
    return v / length(v);
}

template <typename T> HOST DEVICE T maxComponent(const vec<T> &lhs) {
    return lhs.x > lhs.y ? (lhs.x > lhs.z ? lhs.x : lhs.z)
                         : (lhs.y > lhs.z ? lhs.y : lhs.z);
}

template <typename T> HOST DEVICE T minComponent(const vec<T> &lhs) {
    return lhs.x < lhs.y ? (lhs.x < lhs.z ? lhs.x : lhs.z)
                         : (lhs.y < lhs.z ? lhs.y : lhs.z);
}

template <typename S, typename T> struct SameType {
    static constexpr bool value = false;
};

template <typename T> struct SameType<T, T> {
    static constexpr bool value = true;
};

template <typename T> HOST DEVICE vec<T> ceil(const vec<T> &lhs) {
    if constexpr (SameType<T, float>::value) {
        return vec<T>(ceilf(lhs.x), ceilf(lhs.y), ceilf(lhs.z));
    } else if constexpr (SameType<T, double>::value) {
        return vec<T>(::ceil(lhs.x), ::ceil(lhs.y), ::ceil(lhs.z));
    } else {
        return lhs;
    }
}

template <typename T> HOST DEVICE vec<T> floor(const vec<T> &lhs) {
    if constexpr (SameType<T, float>::value) {
        return vec<T>(floorf(lhs.x), floorf(lhs.y), floorf(lhs.z));
    } else if constexpr (SameType<T, double>::value) {
        return vec<T>(::floor(lhs.x), ::floor(lhs.y), ::floor(lhs.z));
    } else {
        return lhs;
    }
}

template <typename T>
HOST DEVICE bool operator==(const vec<T> &lhs, const vec<T> &rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

template <typename T>
HOST DEVICE bool operator!=(const vec<T> &lhs, const vec<T> &rhs) {
    return !(lhs == rhs);
}

typedef vec<float> fvec;
typedef vec<double> dvec;
typedef vec<int32_t> ivec;
typedef vec<uint32_t> uvec;
} // namespace cubble
