
#include "voxel2stl.hpp"
#include <omp.h>

namespace voxel2stl
{
  VoxelData::VoxelData(std::string & filename, int anx, int any, int anz, double am)
    : nx(anx), ny(any), nz(anz), m(am)
  {
    ifstream is(filename, ios::in | ios::binary);
    int num_threads;
#pragma omp parallel
    {
      num_threads = omp_get_num_threads();
    }
    char buffer[nx*ny*nz];
    is.read(buffer,nx*ny*nz);
    if(is)
      cout << "Voxel data read successfully." << endl;
    else
      cout << "Error in reading of voxel data" << endl;
    is.close();
    int nx_old = nx;
    while (! (nx%num_threads))
      nx++;
    data.SetSize(nx*ny*nz);
#pragma omp parallel for
    for(size_t i = 0; i < nx_old*ny*nz; i++)
      data[i] = (double) buffer[i];

    // fill up for parallel computation
#pragma omp parallel for
    for(size_t i = nx_old*ny*nz; i<nx*ny*nz; i++)
      data[i] = 0;
  }

  void VoxelData::WriteMaterials(const string & filename) const
  {
    ofstream of;
    of.open(filename);
    for (auto i : Range(nx))
      for (auto j : Range(ny))
        for (auto k : Range(nz))
          of << i*m << "," << j*m << "," << k*m << "," << (*this)(i,j,k) << endl;
    cout << "Coefficients written to file " << filename << endl;
    of.close();
  }
  
  void VoxelData::WriteMaterials(const string & filename, const Array<int>& region) const
  {
    ofstream of;
    of.open(filename);
    for (auto i : Range(std::max(0,region[0]),std::min(nx,region[1])))
      for (auto j : Range(std::max(0,region[2]),std::min(ny,region[3])))
        for (auto k : Range(std::max(0,region[4]),std::min(nz,region[5])))
          of << i*m << "," << j*m << "," << k*m << "," << (*this)(i,j,k) << endl;
    cout << "Coefficients written to file " << filename << endl;
    of.close();
  }
}
