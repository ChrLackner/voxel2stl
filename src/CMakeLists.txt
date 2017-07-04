
pybind11_add_module(libvoxel2stl.so SHARED
  python_voxel2stl.cpp
  geometry.cpp
  voxeldata.cpp)

install(TARGETS libvoxel2stl.so)
