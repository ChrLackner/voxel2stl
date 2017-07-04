
#include "voxel2stl.hpp"
#include <python_ngstd.hpp>
#include <stdio.h>

using namespace voxel2stl;
using namespace ngfem;

void ExportVoxel2STL(py::module & m)
{
  py::class_<VoxelData,shared_ptr<VoxelData>>
    (m,"VoxelData", "Container for loading of voxel data")
    .def(py::init<string&, int, int, int, double>())
    .def("WriteMaterials", (void (VoxelData::*)(const string&) const) &VoxelData::WriteMaterials,
         "Write the material number of each voxel with it's coordinates to an output file")
    .def("WriteMaterials",[](shared_ptr<VoxelData> self, string & filename, py::list aregion)
         {
           if(py::len(aregion)!=6)
             throw Exception("Must provide 6 integers for region!");
           Array<int> region(6);
           for(auto i : Range(6)) {
             region[i] = py::cast<int>(aregion[i]);
           }
           self->WriteMaterials(filename,region);
         },
         "Write the material number of each voxel restricted to the with it's coordinates to an output file")
    ;

  py::class_<VoxelSTLGeometry>
    (m,"VoxelSTLGeometry", "STLGeometry from voxel data")
    .def("__init__", [](VoxelSTLGeometry* instance,
                        shared_ptr<VoxelData> data, py::object mats, py::object bnds)
         {
           shared_ptr<Array<int>> materials = nullptr;
           shared_ptr<Array<int>> boundaries = nullptr;
           if (!mats.is_none())
             {
               auto matlist = py::cast<py::list>(mats);
               materials = make_shared<Array<int>>(py::len(matlist));
               materials->SetSize(0);
               for (auto mat : matlist)
                 materials->Append(py::cast<int>(mat));
             }
           if (!bnds.is_none())
             {
               auto bndlist = py::cast<py::list>(bnds);
               boundaries = make_shared<Array<int>>(py::len(bndlist));
               boundaries->SetSize(0);
               for (auto bnd : bndlist)
                 boundaries->Append(py::cast<int>(bnd));
             }
           new (instance) VoxelSTLGeometry(data,materials,boundaries);
         }, py::arg("voxeldata"), py::arg("materials")=nullptr, py::arg("boundaries")=nullptr)
    .def("ApplySmoothingStep", &VoxelSTLGeometry::ApplySmoothingStep,
         py::arg("subdivision"),py::arg("weight_area")=0.1,py::arg("weight_minimum")=1.0)
    .def("WriteSTL", &VoxelSTLGeometry::WriteSTL)
    .def("WriteMeshsizeFile", &VoxelSTLGeometry::WriteMeshsizeFile)
    ;
}


PYBIND11_PLUGIN(libvoxel2stl)
{
  py::module m("libvoxel2stl");
  ExportVoxel2STL(m);
  return m.ptr();
}
