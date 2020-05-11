cmake_minimum_required (VERSION 3.0)

project(CUDAInline)

set(LIB_SOURCES
thirdparty/crc64/crc64.cpp
internal/launch_calc.cpp
internal/cuda_wrapper.cpp
internal/nvtrc_wrapper.cpp
Context.cpp
DVBuffer.cpp
DVCombine.cpp
)

set(LIB_HEADERS
DeviceViewable.h
Context.h
DVBuffer.h
DVCombine.h
)

set(INTERNAL_HEADERS
thirdparty/crc64/crc64.h
internal/impl_context.inl
internal/Timing.h
internal/launch_calc.h
internal/cuda_wrapper.h
internal/nvtrc_wrapper.h
)


set (INCLUDE_DIR 
thirdparty/crc64
thirdparty/unqlite
internal
.
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

add_library(unqlite STATIC ../thirdparty/unqlite/unqlite.h ../thirdparty/unqlite/unqlite.c)
add_library(CUDAInline ${LIB_SOURCES} ${LIB_HEADERS} ${INTERNAL_HEADERS})

target_link_libraries(CUDAInline unqlite)


if (WIN32) 
else()
target_link_libraries(CUDAInline dl)
endif()


IF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX  ../install CACHE PATH "Install path" FORCE)
ENDIF(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

install(FILES ${LIB_HEADERS} DESTINATION include)
install(TARGETS CUDAInline DESTINATION lib)

set(BUILD_TESTS false CACHE BOOL "Build tests")

if (BUILD_TESTS)
add_subdirectory(test)
endif()

set(BUILD_PYTHON_BINDINGS true CACHE BOOL "Build Python Bindings")

if (BUILD_PYTHON_BINDINGS)
add_subdirectory(python)
endif()