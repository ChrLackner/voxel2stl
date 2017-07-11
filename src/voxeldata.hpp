#ifndef __FILE_VOXELDATA_HPP
#define __FILE_VOXELDATA_HPP

namespace voxel2stl
{  
  class VoxelData : public pyspdlog::LoggedClass
  {
  private:
    Array<double> data;
    int nx,ny,nz;
    double m;
  
  public:
    VoxelData(std::string & filename, int anx, int any, int anz, double am,
              shared_ptr<spdlog::logger> alog);

    void WriteMaterials(const string & filename) const;
    void WriteMaterials(const string & filename, const Array<int>& region) const;

    double operator()(int i, int j, int k) const { return data[i+nx*(j+ny*k)]; }
    double Getm()  const  { return m; }
    int Getnx() const { return nx; }
    int Getny() const { return ny; }
    int Getnz() const { return nz; }    
  };
}

#endif // __FILE_VOXELDATA_HPP
