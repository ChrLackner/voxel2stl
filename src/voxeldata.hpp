#ifndef __FILE_VOXELDATA_HPP
#define __FILE_VOXELDATA_HPP

namespace voxel2stl
{  
  class VoxelData : public pyspdlog::LoggedClass
  {
  private:
    Array<unsigned char> data;
    std::map<unsigned char, string> material_names;
    size_t nx,ny,nz;
    double m;
  
  public:
    VoxelData(std::string filename, size_t anx, size_t any, size_t anz, double am,
              shared_ptr<spdlog::logger> alog);

    void WriteMaterials(const string & filename) const;
    void WriteMaterials(const string & filename, const Array<size_t>& region) const;

    int operator()(size_t i, size_t j, size_t k) const { return data[i+nx*(j+ny*k)]; }

    std::map<unsigned char, string>& GetMaterialNames() { return material_names; }
    void SetMaterialName(unsigned char nr, string name) { material_names[nr] = name; }

    double Getm()  const  { return m; }
    size_t Getnx() const { return nx; }
    size_t Getny() const { return ny; }
    size_t Getnz() const { return nz; }

    Array<unsigned char>& GetData() { return data; }
  };
}

#endif // __FILE_VOXELDATA_HPP
