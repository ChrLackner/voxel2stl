
#include "voxel2stl.hpp"
#include <stdio.h>
#include <pybind11/pybind11.h>

using namespace voxel2stl;
namespace py = pybind11;

void ExportVoxel2STL(py::module & m)
{
  py::class_<VoxelData,shared_ptr<VoxelData>>
    (m,"VoxelData", "Container for loading of voxel data")
    .def("__init__", [](VoxelData* instance, string& filename, size_t nx, size_t ny, size_t nz,
                        double m, shared_ptr<spdlog::logger> log)
         {
           new (instance) VoxelData(filename, nx, ny, nz, m, log);
         }, py::arg("filename"), py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("m"),
         py::arg("log") = make_shared<spdlog::logger>("VoxelData",make_shared<spdlog::sinks::ansicolor_stdout_sink_mt>()))
    .def("WriteMaterials", (void (VoxelData::*)(const string&) const) &VoxelData::WriteMaterials,
         "Write the material number of each voxel with it's coordinates to an output file")
    .def("WriteMaterials",[](shared_ptr<VoxelData> self, string & filename, py::list aregion)
         {
           if(py::len(aregion)!=6)
             throw Exception("Must provide 6 integers for region!");
           Array<size_t> region(6);
           for(auto i : Range(6)) {
             region[i] = py::cast<size_t>(aregion[i]);
           }
           self->WriteMaterials(filename,region);
         },
         "Write the material number of each voxel restricted to the with it's coordinates to an output file")
    ;

  py::class_<VoxelSTLGeometry, shared_ptr<VoxelSTLGeometry>, pyspdlog::LoggedClass>
    (m,"VoxelSTLGeometry", "STLGeometry from voxel data")
    .def("__init__", [](VoxelSTLGeometry* instance, shared_ptr<VoxelData> data,
                        py::object mats, py::object bnds, shared_ptr<spdlog::logger> log)
         {
           shared_ptr<Array<size_t>> materials = nullptr;
           shared_ptr<Array<size_t>> boundaries = nullptr;
           if (!mats.is_none())
             {
               auto matlist = py::cast<py::list>(mats);
               materials = make_shared<Array<size_t>>(py::len(matlist));
               materials->SetSize(0);
               for (auto mat : matlist)
                 materials->Append(py::cast<size_t>(mat));
             }
           if (!bnds.is_none())
             {
               auto bndlist = py::cast<py::list>(bnds);
               boundaries = make_shared<Array<size_t>>(py::len(bndlist));
               boundaries->SetSize(0);
               for (auto bnd : bndlist)
                 boundaries->Append(py::cast<size_t>(bnd));
             }
           new (instance) VoxelSTLGeometry(data,materials,boundaries,log);
         }, py::arg("voxeldata"), py::arg("materials")=nullptr, py::arg("boundaries")=nullptr,
         py::arg("logger") = make_shared<spdlog::logger>("VoxelSTLGeometry", make_shared<spdlog::sinks::ansicolor_stdout_sink_mt>()))
    .def("SubdivideTriangles", &VoxelSTLGeometry::SubdivideTriangles)
    .def("WriteSTL", &VoxelSTLGeometry::WriteSTL)
    // .def("WriteMeshsizeFile", &VoxelSTLGeometry::WriteMeshsizeFile)
    ;

  py::class_<Smoother, shared_ptr<Smoother>, pyspdlog::LoggedClass>
    (m,"Smoother", "Base class for smoothers")
    .def("Apply", &Smoother::Apply)
    ;

  py::class_<NewtonSmoother<FirstEnergy>, shared_ptr<NewtonSmoother<FirstEnergy>>, Smoother>
    (m,"FirstSmoother", "First simple smoothing alg.")
    .def (py::init<shared_ptr<VoxelSTLGeometry>,shared_ptr<spdlog::logger>>(),
          py::arg("geo"), py::arg("logger") = make_shared<spdlog::logger>
          ("FirstSmoother", make_shared<spdlog::sinks::ansicolor_stdout_sink_mt>()))
    ;
  py::class_<NewtonSmoother<SimpleEnergy>, shared_ptr<NewtonSmoother<SimpleEnergy>>, Smoother>
    (m,"SimpleSmoother", "First simple smoothing alg.")
    .def (py::init<shared_ptr<VoxelSTLGeometry>,shared_ptr<spdlog::logger>>(),
          py::arg("geo"), py::arg("logger") = make_shared<spdlog::logger>
          ("SimpleSmoother", make_shared<spdlog::sinks::ansicolor_stdout_sink_mt>()))
    ;

}


PYBIND11_PLUGIN(libvoxel2stl)
{
  py::module m("libvoxel2stl");
  py::module::import("pyspdlog");
  ExportVoxel2STL(m);
  return m.ptr();
}
