#ifndef __FILE_VOXEL2STL_HPP
#define __FILE_VOXEL2STL_HPP


#include <map>
#include <fem.hpp>
#include "pyspdlog.hpp"
namespace voxel2stl{
  using namespace std;
  inline bool operator==(const Vec<3,double> &x, const Vec<3,double> &y){
     return (abs(x[0]-y[0])<1e-5 && abs(x[1]-y[1])<1e-5 && abs(x[2]-y[2])<1e-5); }
}

#include "voxeldata.hpp"
#include "geometry.hpp"

#endif
