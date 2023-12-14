cmake_minimum_required(VERSION 3.17)

project(cubble LANGUAGES CXX VERSION 0.1.0.0)
set(PROJECT_INSTALL_DIR ${PROJECT_NAME}/v${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}.${PROJECT_VERSION_TWEAK}/${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(EXTERNAL_INCLUDE "{PROJECT_SOURCE_DIR}/include/")

set(gcc "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU,LCC>")
set(msvc "$<COMPILE_LANG_AND_ID:CXX,MSVC>")

set(debug_gcc "$<AND:$<CONFIG:Debug>,${gcc}>")
set(debug_msvc "$<AND:$<CONFIG:Debug>,${msvc}>")

set(gcc_opt -Og;-Wall;-Werror;-Wextra;-Wconversion;-Wsign-conversion;)
set(gcc_opt ${gcc_opt};-pedantic-errors;-Wcast-qual;-Wwrite-strings;)
set(gcc_opt ${gcc_opt};-Wcast-align=strict;-Wparentheses;-Wc++-compat;)
set(gcc_opt ${gcc_opt};-Wlogical-op;-Wlogical-not-parentheses;-Wpadded;)
set(gcc_opt ${gcc_opt};-Wredundant-decls;-Winline;-Winvalid-utf8;-Wformat=2;)
set(gcc_opt ${gcc_opt};-Wformat-security;-Wformat-nonliteral;)
set(gcc_opt ${gcc_opt};-Wnull-dereference;-Winit-self;-Wuninitialized;)
set(gcc_opt ${gcc_opt};-Warith-conversion;-Wduplicated-branches;)
set(gcc_opt ${gcc_opt};-Wshadow;-Wpointer-arith;-Wundef;-Wbad-function-cast;)
set(gcc_opt ${gcc_opt};-Wduplicated-cond;-Wformat-signedness;)

set(msvc_opt -W3;)

add_compile_options(
    "$<${debug_gcc}:${gcc_opt}>"
    "$<${debug_msvc}:${msvc_opt}>"
    )

# TODO: add tests
#option(PACKAGE_TESTS "Build tests" ON)
#if(PACKAGE_TESTS)
#    include(CTest)
#    add_subdirectory(tests)
#    enable_testing()
#endif()

add_subdirectory(src)
