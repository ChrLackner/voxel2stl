
pybind11_add_module(voxel2stl SHARED
  python_voxel2stl.cpp
  geometry.cpp
  voxeldata.cpp
  smoother.cpp
  )

target_link_libraries(voxel2stl PRIVATE ngs_core)

set_target_properties(voxel2stl PROPERTIES
  PREFIX "lib"
  SUFFIX ".so"
  INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS voxel2stl DESTINATION voxel2stl)
