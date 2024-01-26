cmake_minimum_required(VERSION 3.17)

option(remove_asserts "Remove assertions from the code" OFF)
option(enable_profiling "Enable profiling the code" OFF)

add_compile_definitions("$<$<CONFIG:Debug>:CUBBLE_DEBUG>")

set(objs objs)
set(static_library ${PROJECT_NAME})
set(binary ${PROJECT_NAME}-cli)

add_library(${static_library} STATIC)
add_library(${objs} OBJECT "")
add_executable(${binary})

target_sources(
    ${objs}
    PRIVATE
    device_globals.cu
    free_functions.cu
    kernels.cu
    parsing.cpp
    simulation_parameters.cpp
    simulator.cu
    )

target_sources(
    ${binary}
    PRIVATE
    main.cpp
    )

include_directories(PUBLIC "${PROJECT_BINARY_DIR}/src")
include_directories(PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
include_directories(SYSTEM PRIVATE
    "${external_include}"
    )

set_target_properties(${objs} PROPERTIES CUDA_ARCHITECTURES "${cuda_archs}")
set_target_properties(${static_library} PROPERTIES CUDA_ARCHITECTURES "${cuda_archs}")
set_target_properties(${binary} PROPERTIES CUDA_ARCHITECTURES "${cuda_archs}")

target_link_libraries(${static_library} PUBLIC ${objs} nvToolsExt pthread curand)
target_link_libraries(${binary} PUBLIC ${static_library})

install(
    TARGETS ${static_library} ${binary}
    CONFIGURATIONS Release RelWithDebInfo Debug
    ARCHIVE DESTINATION ${project_install_dir}/lib
    LIBRARY DESTINATION ${project_install_dir}/lib
    RUNTIME DESTINATION ${project_install_dir}/bin
    PUBLIC_HEADER DESTINATION ${project_install_dir}/include
    )
