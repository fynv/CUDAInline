cmake_minimum_required (VERSION 3.0)

add_executable(test test.cpp)

target_link_libraries(test CUDAInline)

install(TARGETS test RUNTIME DESTINATION test_cpp)
