
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
    char* buffer = (char*) malloc(nx*ny*nz);
    is.read(buffer,nx*ny*nz);
    // log->debug(std::string(buffer));
    log->flush();
    if(is)
      log->info("Voxel data read successfully.");
    else
      log->error("Error in reading of voxel data from file " + filename);
    log->flush();
    is.close();
    data.SetSize(nx*ny*nz);
#pragma omp parallel for
    for(size_t i = 0; i < nx*ny*nz; i++)
      data[i] = (double) buffer[i];

    free(buffer);
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
