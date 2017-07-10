

cmake_minimum_required(VERSION 3.0)

project(voxel2stl)
set(voxel2stl_VERSION 0.1)

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

# Python
find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)

# pyspdlog
find_package(Pyspdlog REQUIRED)
include_directories(${PYSPDLOG_INCLUDE_DIRS})
include_directories(${PYBIND11_INCLUDE_DIR})

find_package(NGSolve REQUIRED
  HINTS $ENV{NETGENDIR}/.. /opt/netgen/ /Applications/Netgen.app/Contents/Resources/CMake C:/netgen
)
include_directories(${NGSOLVE_INCLUDE_DIRS})

add_subdirectory(external_dependencies/pybind11)
add_subdirectory(src)
add_subdirectory(python)

# Tests
include(CTest)
add_subdirectory(tests)
