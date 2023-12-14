cmake_minimum_required(VERSION 3.17)

option(REMOVE_ASSERTS "Remove assertions from the code" OFF)
option(ENABLE_PROFILING "Enable profiling the code" OFF)

configure_file(config.h.in config.h @ONLY)

add_compile_definitions("$<$<CONFIG:Debug>:CUBBLE_DEBUG>")

set(OBJS cpp_objs)
set(STATIC_LIBRARY ${PROJECT_NAME})

add_library(${STATIC_LIBRARY} STATIC)
add_library(${OBJS} OBJECT "")

target_sources(
    ${OBJS}
    PRIVATE
    main.cpp
    simulator.cu
    kernels.cu
    )

include_directories(PUBLIC "${PROJECT_BINARY_DIR}/src")
include_directories(PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
include_directories(SYSTEM PRIVATE
    "${EXTERNAL_INCLUDE}"
    )

# TODO precompiled flags didn't work with cuda, right?
target_precompile_headers(
    ${OBJS}
    PRIVATE
    )
# TODO add cuda, check if these are needed
target_link_libraries(${STATIC_LIBRARY} PUBLIC ${OBJS} cudart curand nvToolsExt pthread)

install(
    TARGETS ${STATIC_LIBRARY}
    CONFIGURATIONS Release RelWithDebInfo Debug
    ARCHIVE DESTINATION ${PROJECT_INSTALL_DIR}/lib
    LIBRARY DESTINATION ${PROJECT_INSTALL_DIR}/lib
    RUNTIME DESTINATION ${PROJECT_INSTALL_DIR}/bin
    PUBLIC_HEADER DESTINATION ${PROJECT_INSTALL_DIR}/include
    )
