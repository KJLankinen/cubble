cmake_minimum_required(VERSION 3.17)

project(cubble LANGUAGES CXX CUDA VERSION 0.1.0.0)
set(PROJECT_INSTALL_DIR ${PROJECT_NAME}/v${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}.${PROJECT_VERSION_TWEAK}/${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

set(EXTERNAL_INCLUDE "${PROJECT_SOURCE_DIR}/include/")

set(gcc "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")

set(debug_gcc "$<AND:$<CONFIG:Debug>,${gcc}>")

set(gcc_warn -Wall;-Werror;-Wextra;-Wconversion;-Wsign-conversion;)
set(gcc_warn ${gcc_warn};-pedantic-errors;-Wcast-qual;-Wwrite-strings;)
set(gcc_warn ${gcc_warn};-Wcast-align=strict;-Wparentheses;)
set(gcc_warn ${gcc_warn};-Wlogical-op;-Wlogical-not-parentheses;-Wpadded;)
set(gcc_warn ${gcc_warn};-Wredundant-decls;-Winline;-Wformat=2;)
set(gcc_warn ${gcc_warn};-Wformat-security;-Wformat-nonliteral;)
set(gcc_warn ${gcc_warn};-Wnull-dereference;-Winit-self;-Wuninitialized;)
set(gcc_warn ${gcc_warn};-Warith-conversion;-Wduplicated-branches;)
set(gcc_warn ${gcc_warn};-Wshadow;-Wpointer-arith;-Wundef;)
set(gcc_warn ${gcc_warn};-Wduplicated-cond;-Wformat-signedness;)

set(gcc_deb_opt -Og;)
set(gcc_rel_opt -O3;)

set(nvcc "$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>")

add_compile_options(
    "$<${gcc}:${gcc_warn}>"
    "$<IF:${debug_gcc},${gcc_deb_opt},${gcc_rel_opt}>"
    )

# TODO: add tests
#option(PACKAGE_TESTS "Build tests" ON)
#if(PACKAGE_TESTS)
#    include(CTest)
#    add_subdirectory(tests)
#    enable_testing()
#endif()

add_subdirectory(src)
