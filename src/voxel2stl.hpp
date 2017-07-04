#ifndef __FILE_VOXEL2STL_HPP
#define __FILE_VOXEL2STL_HPP

#include <bla.hpp>
using namespace ngbla;

#include <map>
#include "pyspdlog.hpp"
namespace voxel2stl{

  inline bool operator==(const Vec<3,double> &x, const Vec<3,double> &y){
    return (abs(x[0]-y[0])<1e-5 && abs(x[1]-y[1])<1e-5 && abs(x[2]-y[2])<1e-5); }
}

#include "voxeldata.hpp"
#include "geometry.hpp"
#include <fem.hpp>
#include "gridcoefficientfunction.hpp"



#endif
