/*
    Cubble
    Copyright (C) 2024  Juhana Lankinen

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

#include <string>

// This is wasteful in terms of space, but simple
namespace cubble {
template <typename T>
struct Result {
    T t = {};
    const std::string msg = "";
    const bool ok = false;
    uint8_t padding[7] = {};

  public:
    Result<T>(T ty, const std::string &m, bool success)
        : t(ty), msg(m), ok(success) {}
    static Result<T> Ok(const T &t) { return Result<T>{t, "", true}; }
    static Result<T> Err(const std::string &msg) {
        return Result<T>{T{}, msg, false};
    }
};
} // namespace cubble
