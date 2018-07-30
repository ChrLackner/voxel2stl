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

    VoxelData(Array<unsigned char>& adata, size_t anx, size_t any, size_t anz, double am,
              shared_ptr<spdlog::logger> alog)
      : nx(anx), ny(any), nz(anz), m(am), LoggedClass(alog)
    {
      data.Swap(adata);
      for(auto i : Range(nx*ny*nz))
        if(data[i])
          if(!material_names.count(data[i]))
            material_names[data[i]] = "mat" + std::to_string(data[i]);
    }

    void WriteMaterials(const string & filename) const;
    void WriteMaterials(const string & filename, const Array<size_t>& region) const;

    int operator()(size_t i, size_t j, size_t k) const { return data[i+nx*(j+ny*k)]; }

    std::map<unsigned char, string>& GetMaterialNames() { return material_names; }
    void SetMaterialName(unsigned char nr, string name) { material_names[nr] = name; }

    double Getm()  const  { return m; }
    size_t Getnx() const { return nx; }
    size_t Getny() const { return ny; }
    size_t Getnz() const { return nz; }

    void EmbedInAir()
    {
      nx += 2; ny += 2; nz += 2;
      Array<unsigned char> newdata(nx*ny*nz);
      for(auto z : Range(nz))
        for(auto y : Range(ny))
          for(auto x : Range(nx))
            if(x == 0 || y == 0 || z == 0 || x == nx-1 || y == ny-1 || z == nz-1)
              newdata[x+nx*(y+ny*z)] = 0;
            else
              newdata[x+nx*(y+ny*z)] = data[(x-1) + (nx-2)*((y-1) + (ny-2) * (z-1))];
      data.Swap(newdata);
    }

    Array<unsigned char>& GetData() { return data; }
  };
}

#endif // __FILE_VOXELDATA_HPP
