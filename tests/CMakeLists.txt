cmake_minimum_required(VERSION 3.17)

set(testname cubble_tests)
# create the testing file and list of tests
create_test_sourcelist(Tests
  ${testname}.cpp
  first_test.cpp
  second_test.cpp
  )

# add the executable
add_executable(${testname} ${Tests})

include_directories(PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(${testname} PUBLIC ${PROJECT_NAME})
set_target_properties(${testname} PROPERTIES CUDA_ARCHITECTURES "${cuda_archs}")

# remove the test driver source file
set(TestsToRun ${Tests})
remove(TestsToRun ${testname}.cpp)

# Add all the ADD_TEST for each test
foreach(test ${TestsToRun})
  get_filename_component(TName ${test} NAME_WE)
  add_test(NAME ${TName} COMMAND ${testname} ${TName})
endforeach()
