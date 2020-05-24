cmake_minimum_required (VERSION 3.0)

project(PyCUDAInline)

add_custom_target(Run_CFFIBuild
COMMAND python CUDAInline/cffi_build.py
WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
COMMENT "Running cffi_build")

set (SRC
api.h
api_utils.cpp
api_Context.cpp
api_DeviceViewable.cpp
api_DVBuffer.cpp
api_DVCombine.cpp
)

set (INCLUDE_DIR 
.
..
)

if (WIN32) 
set (DEFINES  ${DEFINES}
-D"_CRT_SECURE_NO_DEPRECATE"  
-D"_SCL_SECURE_NO_DEPRECATE" 
-D"_CRT_SECURE_NO_WARNINGS"
)
else()
add_definitions(-std=c++14)
add_compile_options(-fPIC)
endif()

include_directories(${INCLUDE_DIR})
add_definitions(${DEFINES})

add_library(PyCUDAInline SHARED ${SRC})
target_link_libraries(PyCUDAInline CUDAInline)


if (WIN32) 
install(TARGETS PyCUDAInline RUNTIME DESTINATION test_python/CUDAInline)
else()
install(TARGETS PyCUDAInline DESTINATION test_python/CUDAInline)
endif()



set(PYTHON
	CUDAInline/__init__.py
	CUDAInline/cffi.py
	CUDAInline/Native.py
	CUDAInline/utils.py
	CUDAInline/Context.py
	CUDAInline/DeviceViewable.py
	CUDAInline/DVBuffer.py
	CUDAInline/DVCombine.py
	CUDAInline/DVVector.py
)

install(FILES ${PYTHON} DESTINATION test_python/CUDAInline)
install(FILES setup.py README.md DESTINATION test_python)

set(INCLUDE_TESTS false CACHE BOOL "Include tests")

if (INCLUDE_TESTS)
add_subdirectory(test)
endif()

