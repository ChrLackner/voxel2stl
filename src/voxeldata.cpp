
#include "voxel2stl.hpp"
#include <omp.h>

namespace voxel2stl
{
  VoxelData::VoxelData(std::string filename, size_t anx, size_t any, size_t anz, double am,
                       shared_ptr<spdlog::logger> alog)
    : nx(anx), ny(any), nz(anz), m(am), LoggedClass(alog)
  {
    log->debug("Voxeldata size is " + to_string(nx*ny*nz));
    ifstream is(filename, ios::in | ios::binary);
    if (!is)
      {
        log->error("Voxel data file '" + filename + "' couldn't be loaded!");
        log->flush();
        throw exception();
      }
    int num_threads;
#pragma omp parallel
    {
      num_threads = omp_get_num_threads();
    }
    auto nx_old = nx;
    while (! (nx%num_threads))
      nx++;
    data.SetSize(nx*ny*nz);
    is.read((char*) &data[0],nx_old*ny*nz);
    if(is)
      log->info("Voxel data read successfully.");
    else
      log->error("Error in reading of voxel data from file " + filename);
    log->flush();
    is.close();

    // fill up for parallel computation
#pragma omp parallel for
    for(size_t i = nx_old*ny*nz; i<nx*ny*nz; i++)
      data[i] = 0;

    for(auto i : Range(nx*ny*nz))
      if(data[i])
        if(!material_names.count(data[i]))
          material_names[data[i]] = "mat" + std::to_string(data[i]);
  }

  void VoxelData::WriteMaterials(const string & filename) const
  {
    log->debug("Start writing materials");
    log->flush();
    ofstream of;
    of.open(filename);
    for (auto i : Range(nx))
      for (auto j : Range(ny))
        for (auto k : Range(nz))
          of << i*m << "," << j*m << "," << k*m << "," << (*this)(i,j,k) << endl;
    log->info("Coefficients written to file " + filename);
    of.close();
  }
  
  void VoxelData::WriteMaterials(const string & filename, const Array<size_t>& region) const
  {
    log->debug("Start writing materials of regions:");
    log->flush();
    for(auto i : region)
      log->debug(i);
    ofstream of;
    of.open(filename);
    for (auto i : Range(std::max(size_t(0),region[0]),std::min(nx,region[1])))
      for (auto j : Range(std::max(size_t(0),region[2]),std::min(ny,region[3])))
        for (auto k : Range(std::max(size_t(0),region[4]),std::min(nz,region[5])))
          of << i*m << "," << j*m << "," << k*m << "," << (*this)(i,j,k) << endl;
    log->info("Coefficients written to file " + filename);
    of.close();
  }
}
