
#include "voxel2stl.hpp"
#include <stdio.h>
#include <python_ngstd.hpp>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace voxel2stl;


template<typename T>
py::object MoveToNumpyArray( ngstd::Array<T> &a )
{
  if(a.Size()) {
      py::capsule free_when_done(&a[0], [](void *f) {
                                 delete [] reinterpret_cast<T *>(f);
                                 });
      a.NothingToDelete();
      return py::array_t<T>(a.Size(), &a[0], free_when_done);
  }
  else
      return py::array_t<T>(0, nullptr);
}

void ExportVoxel2STL(py::module & m)
{
  py::class_<VoxelData,shared_ptr<VoxelData>>
    (m,"VoxelData", "Container for loading of voxel data")
    .def(py::init<string, size_t, size_t, size_t, double, shared_ptr<spdlog::logger>>(),
         py::arg("filename"), py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("m"),
         py::arg("log") = make_shared<spdlog::logger>("VoxelData",make_shared<spdlog::sinks::ansicolor_stdout_sink_mt>()))
    .def(py::pickle([](VoxelData& self)
                    {
                      MemoryView mv(&self.GetData()[0], self.GetData().Size());
                      return py::make_tuple(self.Getnx(), self.Getny(), self.Getnz(), self.Getm(),
                                            mv, self.GetMaterialNames());
                    },
                    [](py::tuple state)
                    {
                      auto mv = py::cast<MemoryView>(state[4]);
                      Array<unsigned char> data(mv.Size(),(unsigned char*) mv.Ptr(),true);
                      VoxelData voxel(data, state[0].cast<size_t>(), state[1].cast<size_t>(),
                                      state[2].cast<size_t>(), state[3].cast<double>(),
                                      make_shared<spdlog::logger>("VoxelData", make_shared<spdlog::sinks::ansicolor_stdout_sink_mt>()));
                      auto mats = state[5].cast<map<unsigned char, string>>();
                      for (auto val : mats)
                        voxel.SetMaterialName(val.first, val.second);
                      return voxel;
                    }))
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
    .def_property_readonly("dims", [] (VoxelData& self)
                           {
                             return py::make_tuple(self.Getnx(),
                                                   self.Getny(),
                                                   self.Getnz());
                           })
    .def_property_readonly("voxelsize", &VoxelData::Getm)
    .def("GetBoundingBox", [](VoxelData& self)
         {
           Vec<3> size = {self.Getnx() * self.Getm(),
                               self.Getny() * self.Getm(),
                               self.Getnz() * self.Getm()};
           Vec<3> bot = - 0.5 * size;
           Vec<3> top = 1.5 * size;
           auto pybot = py::make_tuple(bot[0],bot[1],bot[2]);
           auto pytop = py::make_tuple(top[0],top[1],top[2]);
           return py::make_tuple(pybot, pytop);
         })
    .def("GetMaterialNames", &VoxelData::GetMaterialNames, py::return_value_policy::reference_internal)
    .def("SetMaterialNames", [](VoxelData& self, py::list matnames)
         {
           for(auto i : Range(py::len(matnames)))
             if(self.GetMaterialNames().count(i+1) == 0)
               self.GetLogger()->warn("Material " + py::cast<string>(matnames[i]) + " not used!");
           else
             self.SetMaterialName((unsigned char) (i+1),py::cast<string>(matnames[i]));
         })
    .def_property_readonly("data", [](VoxelData& self)
                           {
                             return py::array_t<unsigned char>(self.GetData().Size(),
                                                               &self.GetData()[0],
                                                               nullptr);
                           }, py::return_value_policy::reference_internal)
    ;

  py::class_<VoxelSTLGeometry, shared_ptr<VoxelSTLGeometry>, pyspdlog::LoggedClass>
    (m,"VoxelSTLGeometry", "STLGeometry from voxel data", py::dynamic_attr())
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
    .def("GetData", [](VoxelSTLGeometry& self)
         {
           Array<float> verts;
           verts.SetAllocSize(self.GetNTriangles()*9);
           Array<unsigned int> trigs;
           trigs.SetAllocSize(self.GetNTriangles()*4);
           Array<float> normals;
           normals.SetAllocSize(self.GetNTriangles()*9);
           for(auto i : Range(self.GetNTriangles()))
             {
               const auto& trig = self.GetTriangle(i);
               for(auto v : { trig.v1, trig.v2, trig.v3 })
                 {
                   for(auto j : Range(3))
                     {
                       verts.Append(v->xy[j]*self.GetVoxelSize());
                       normals.Append(trig.n[j]);
                     }
                 }
               for(auto j : Range(3))
                 trigs.Append(i*3+j);
               trigs.Append(0);
             }
           py::dict data;
           data["vertices"] = MoveToNumpyArray(verts);
           data["triangles"] = MoveToNumpyArray(trigs);
           data["normals"] = MoveToNumpyArray(normals);
           return data;
         }, "data for visualization in new gui")
    .def("GetBoundingBox", [](VoxelSTLGeometry& self)
         {
           Array<double> bounds;
           if(self.GetBoundaries())
             for (auto val : *self.GetBoundaries())
               bounds.Append(self.GetVoxelSize() * val);
           else
             for(auto val : {self.GetVoxelData()->Getnx(),
                   self.GetVoxelData()->Getny(),
                   self.GetVoxelData()->Getnz()})
               {
                 bounds.Append(0);
                 bounds.Append(val*self.GetVoxelSize());
               }
           return MoveToNumpyArray(bounds);
         })
    ;

  py::class_<Smoother, shared_ptr<Smoother>, pyspdlog::LoggedClass>
    (m,"Smoother", "Base class for smoothers")
    .def("Apply", &Smoother::Apply,py::call_guard<py::gil_scoped_release>())
    .def("CalcEnergy", [](Smoother& self)
         {
           auto energy = self.CalcEnergy();
           {
             py::gil_scoped_acquire ac;
             return MoveToNumpyArray(energy);
           }
         },py::call_guard<py::gil_scoped_release>())
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
  py::class_<NewtonSmoother<AngleDeficit>, shared_ptr<NewtonSmoother<AngleDeficit>>, Smoother>
    (m,"AngleDeficitSmoother", "from http://www.cc.ac.cn/06research_report/0601.pdf")
    .def (py::init<shared_ptr<VoxelSTLGeometry>,shared_ptr<spdlog::logger>>(),
          py::arg("geo"), py::arg("logger") = make_shared<spdlog::logger>
          ("AngleDeficitSmoother", make_shared<spdlog::sinks::ansicolor_stdout_sink_mt>()))
    ;
}


PYBIND11_PLUGIN(libvoxel2stl)
{
  py::module m("libvoxel2stl");
  py::module::import("pyspdlog");
  ExportVoxel2STL(m);
  return m.ptr();
}
