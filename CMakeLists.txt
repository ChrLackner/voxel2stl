

cmake_minimum_required(VERSION 3.0)

project(voxel2stl)
set(voxel2stl_VERSION 0.1)

find_package(NGSolve CONFIG REQUIRED
  HINTS $ENV{NETGENDIR}/.. /opt/netgen/ /Applications/Netgen.app/Contents/Resources/CMake C:/netgen
  )

option(TEST_GEOMETRY "Build additional geometry for testing")

# ccache
if(USE_CCACHE)
  find_program(CCACHE_FOUND ccache)
  if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
  endif(CCACHE_FOUND)
endif(USE_CCACHE)

# OpenMP
find_package(OpenMP)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# pyspdlog
find_package(Pyspdlog REQUIRED)
include_directories(${PYSPDLOG_INCLUDE_DIRS})
include_directories(${PYBIND11_INCLUDE_DIR})

add_subdirectory(src)
add_subdirectory(python)

# Tests
include(CTest)
add_subdirectory(tests)
