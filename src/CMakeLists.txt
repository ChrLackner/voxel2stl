
pybind11_add_module(voxel2stl SHARED
  python_voxel2stl.cpp
  geometry.cpp
  voxeldata.cpp)

set_target_properties(voxel2stl PROPERTIES
  PREFIX "lib"
  SUFFIX ".so"
  INSTALL_RPATH_USE_LINK_PATH TRUE)

target_link_libraries(voxel2stl PUBLIC ngsolve ${PYSPDLOG_LIBRARIES})

install(TARGETS voxel2stl DESTINATION voxel2stl)
