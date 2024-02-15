cmake_minimum_required(VERSION 3.17)

project(cubble LANGUAGES CXX CUDA VERSION 0.1.0.0)
set(project_install_dir ${PROJECT_NAME}/v${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}.${PROJECT_VERSION_TWEAK}/${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(external_include "${PROJECT_SOURCE_DIR}/include/")

set(gcc "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")

set(debug_gcc "$<AND:$<CONFIG:Debug>,${gcc}>")

set(gcc_warn -Wall;-Werror;-Wextra;-Wconversion;-Wsign-conversion;)
set(gcc_warn ${gcc_warn};-pedantic-errors;-Wcast-qual;-Wwrite-strings;)
set(gcc_warn ${gcc_warn};-Wcast-align=strict;-Wparentheses;)
set(gcc_warn ${gcc_warn};-Wlogical-op;-Wlogical-not-parentheses;)
set(gcc_warn ${gcc_warn};-Wredundant-decls;-Wformat=2;)
set(gcc_warn ${gcc_warn};-Wformat-security;-Wformat-nonliteral;)
set(gcc_warn ${gcc_warn};-Wnull-dereference;-Winit-self;-Wuninitialized;)
set(gcc_warn ${gcc_warn};-Warith-conversion;-Wduplicated-branches;)
set(gcc_warn ${gcc_warn};-Wpointer-arith;-Wundef;)
set(gcc_warn ${gcc_warn};-Wduplicated-cond;-Wformat-signedness;)

set(gcc_deb_opt -Og;)
set(gcc_rel_opt -O3;)

set(nvcc "$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>")
set(cuda_archs 70;72)

add_compile_options(
    "$<${gcc}:${gcc_warn}>"
    "$<IF:${debug_gcc},${gcc_deb_opt},${gcc_rel_opt}>"
    )

option(package_tests "Build tests" ON)
if(package_tests)
    include(CTest)
    add_subdirectory(tests)
    enable_testing()
endif()

add_subdirectory(src)
