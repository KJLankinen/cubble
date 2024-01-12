cmake_minimum_required(VERSION 3.17)

option(REMOVE_ASSERTS "Remove assertions from the code" OFF)
option(ENABLE_PROFILING "Enable profiling the code" OFF)

add_compile_definitions("$<$<CONFIG:Debug>:CUBBLE_DEBUG>")

set(OBJS objs)
set(STATIC_LIBRARY ${PROJECT_NAME})
set(BINARY ${PROJECT_NAME}-cli)

add_library(${STATIC_LIBRARY} STATIC)
add_library(${OBJS} OBJECT "")
add_executable(${BINARY})

target_sources(
    ${OBJS}
    PRIVATE
    simulator.cu
    kernels.cu
    )

target_sources(
    ${BINARY}
    PRIVATE
    main.cpp
    )

include_directories(PUBLIC "${PROJECT_BINARY_DIR}/src")
include_directories(PRIVATE "${CMAKE_CURRENT_LIST_DIR}")
include_directories(SYSTEM PRIVATE
    "${EXTERNAL_INCLUDE}"
    )

set(CUDA_ARCHS 70;72)
set_target_properties(${OBJS} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCHS}")
set_target_properties(${STATIC_LIBRARY} PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCHS}")

target_link_libraries(${STATIC_LIBRARY} PUBLIC ${OBJS} nvToolsExt pthread curand)
target_link_libraries(${BINARY} PUBLIC ${STATIC_LIBRARY})

install(
    TARGETS ${STATIC_LIBRARY} ${BINARY}
    CONFIGURATIONS Release RelWithDebInfo Debug
    ARCHIVE DESTINATION ${PROJECT_INSTALL_DIR}/lib
    LIBRARY DESTINATION ${PROJECT_INSTALL_DIR}/lib
    RUNTIME DESTINATION ${PROJECT_INSTALL_DIR}/bin
    PUBLIC_HEADER DESTINATION ${PROJECT_INSTALL_DIR}/include
    )
