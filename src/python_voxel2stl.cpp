
#include "voxel2stl.hpp"
#include <stdio.h>
#include <python_ngstd.hpp>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#ifdef TEST_GEOMETRY
#include "test_geometry.hpp"
#endif // TEST_GEOMETRY

using namespace voxel2stl;

shared_ptr<spdlog::sinks::sink> global_sink = make_shared<spdlog::sinks::ansicolor_stdout_sink_mt>();

shared_ptr<spdlog::logger> CreateLogger(string name)
{
  auto logger = make_shared<spdlog::logger>(name, global_sink);
  logger->set_level(spdlog::level::debug);
  return logger;
}

void ExportVoxel2STL(py::module & m)
{
  global_sink->set_level(spdlog::level::info);
  m.def("SetGlobalSinkLevel", [](spdlog::level::level_enum en)
        {
          global_sink->set_level(en);
        });
  py::class_<VoxelData,shared_ptr<VoxelData>>
    (m,"VoxelData", "Container for loading of voxel data")
    .def(py::init<string, size_t, size_t, size_t, double, shared_ptr<spdlog::logger>>(),
         py::arg("filename"), py::arg("nx"), py::arg("ny"), py::arg("nz"), py::arg("m"),
         py::arg("log") = CreateLogger("Voxeldata"))
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
    .def("EmbedInAir", &VoxelData::EmbedInAir)
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
           Vec<3> bot = 0; // -0.5*self.Getm();
           Vec<3> top = 2*size;
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
    .def(py::init([](shared_ptr<VoxelData> data,
                     py::object mats, py::object bnds, shared_ptr<spdlog::logger> log)
         {
           Array<size_t> materials;
           Array<size_t> boundaries;
           {
             py::gil_scoped_acquire ac;
             if (!mats.is_none())
               for (auto mat : py::cast<py::list>(mats))
                 materials.Append(py::cast<size_t>(mat));
             if (!bnds.is_none())
               for (auto bnd : py::cast<py::list>(bnds))
               boundaries.Append(py::cast<size_t>(bnd));
           }
           return VoxelSTLGeometry(data,materials,boundaries,log);
         }), py::arg("voxeldata"), py::arg("materials")=nullptr, py::arg("boundaries")=nullptr,
         py::arg("logger") = CreateLogger("VoxelSTLGeometry"), py::call_guard<py::gil_scoped_release>())
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
           if(self.GetBoundaries().Size())
             for (auto val : self.GetBoundaries())
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
    .def("CalcVertexEnergy", [](Smoother& self, size_t index)
         {
           return self.CalcVertexEnergy(index);
         })
        ;

  py::class_<NewtonSmoother<FirstEnergy>, shared_ptr<NewtonSmoother<FirstEnergy>>, Smoother>
    (m,"FirstSmoother", "First simple smoothing alg.")
    .def (py::init<shared_ptr<VoxelSTLGeometry>,shared_ptr<spdlog::logger>>(),
          py::arg("geo"), py::arg("logger") = CreateLogger("FirstSmoother"))
    ;
  py::class_<NewtonSmoother<SimpleEnergy>, shared_ptr<NewtonSmoother<SimpleEnergy>>, Smoother>
    (m,"SimpleSmoother", "First simple smoothing alg.")
    .def (py::init<shared_ptr<VoxelSTLGeometry>,shared_ptr<spdlog::logger>>(),
          py::arg("geo"), py::arg("logger") = CreateLogger("SimpleSmoother"))
    ;
  py::class_<NewtonSmoother<PatchEnergy<AngleDeficit>>, shared_ptr<NewtonSmoother<PatchEnergy<AngleDeficit>>>, Smoother>
    (m,"AngleDeficitSmoother", "from http://www.cc.ac.cn/06research_report/0601.pdf")
    .def (py::init<shared_ptr<VoxelSTLGeometry>,shared_ptr<spdlog::logger>>(),
          py::arg("geo"), py::arg("logger") = CreateLogger("AngleDeficitSmoother"))
    ;
  py::class_<NewtonSmoother<PatchEnergy<TaubinIntegralFormulation>>, shared_ptr<NewtonSmoother<PatchEnergy<TaubinIntegralFormulation>>>,
             Smoother>
    (m,"TaubinSmoother")
    .def (py::init<shared_ptr<VoxelSTLGeometry>,shared_ptr<spdlog::logger>>(),
          py::arg("geo"), py::arg("logger") = CreateLogger("TaubinSmoother"))
    ;
  py::class_<NewtonSmoother<PatchEnergy<CurvatureEstimation>>, shared_ptr<NewtonSmoother<PatchEnergy<CurvatureEstimation>>>,
             Smoother>
    (m,"CurvatureSmoother")
    .def (py::init<shared_ptr<VoxelSTLGeometry>,shared_ptr<spdlog::logger>>(),
          py::arg("geo"), py::arg("logger") = CreateLogger("CurvatureSmoother"))
    ;
  py::class_<NewtonSmoother<WeightedEnergy<FirstEnergy,TaubinIntegralFormulation,70>>,
             shared_ptr<NewtonSmoother<WeightedEnergy<FirstEnergy,TaubinIntegralFormulation,70>>>,
             Smoother>
    (m,"FirstAndTaubinSmoother")
    .def (py::init<shared_ptr<VoxelSTLGeometry>,shared_ptr<spdlog::logger>>(),
          py::arg("geo"), py::arg("logger") = CreateLogger("FirstAndTaubinSmoother"))
    ;
  m.def("GetClosestVertex", [](VoxelSTLGeometry& geo, py::tuple pnt)
        {
          Vec<3> p = {pnt[0].cast<double>(), pnt[1].cast<double>(), pnt[2].cast<double>()};
          auto vertex = geo.GetClosestVertex(p);
          return make_tuple(vertex->xy[0], vertex->xy[1], vertex->xy[2]);
        });
  m.def("GetVertexPatchArea", [](VoxelSTLGeometry& geo, py::tuple pnt)
        {
          Vec<3> p = {pnt[0].cast<double>(), pnt[1].cast<double>(), pnt[2].cast<double>()};
          auto vertex = geo.GetClosestVertex(p);
          double area = 0;
          for(auto trig : vertex->neighbours)
            area += trig->area;
          return area;
        });
  m.def("GetVertexNumber", [](VoxelSTLGeometry& geo, py::tuple pnt)
        {
          Vec<3> p = {pnt[0].cast<double>(), pnt[1].cast<double>(), pnt[2].cast<double>()};
          auto vertex = geo.GetVertex(0);
          size_t index = 0;
          for (auto i : Range(geo.GetNVertices()))
            if (L2Norm(geo.GetVoxelSize() * geo.GetVertex(i)->xy - p) < L2Norm(geo.GetVoxelSize() * vertex->xy - p))
              {
                vertex = geo.GetVertex(i);
                index = i;
              }
          return index;
        });
  m.def("GetVertexRatio", [](VoxelSTLGeometry& geo, size_t index)
        {
          return geo.GetVertex(index)->ratio;
        });
  m.def("SetVertexRatio", [](VoxelSTLGeometry& geo, size_t index, double ratio)
        {
          geo.GetVertex(index)->SetRatio(ratio);
        });
  m.def("GetEigenValues", [](VoxelSTLGeometry& geo, py::tuple pnt)
        {
          Vec<3> p = {pnt[0].cast<double>(), pnt[1].cast<double>(), pnt[2].cast<double>()};
          auto vertex = geo.GetClosestVertex(p);
          auto evals = CurvatureEstimation::EigenValues(vertex);
          return py::make_tuple(evals[0],evals[1], evals[2]);
          // Vec<3> p = {pnt[0].cast<double>(), pnt[1].cast<double>(), pnt[2].cast<double>()};
          // auto vertex = geo.GetClosestVertex(p);
          // std::map<VoxelSTLGeometry::Vertex*,double> patch_w;
          // Vec<3> normal = 0;
          // for(auto trig : vertex->neighbours)
          //   {
          //     normal += trig->area * trig->n;
          //     patch_w[trig->OtherVertices(vertex,0)] += trig->area;
          //     patch_w[trig->OtherVertices(vertex,1)] += trig->area;
          //     patch_w[trig->OtherVertices(vertex,0)] = 1;
          //     patch_w[trig->OtherVertices(vertex,1)] = 1;
          //   }
          // normal /= L2Norm(normal);
          // double normw = 0;
          // for (auto it : patch_w)
          //   normw += it.second*it.second;
          // double fac = 1./sqrt(normw);
          // for(auto it : patch_w)
          //   patch_w[it.first] *= fac;
          // Mat<3> M;
          // M = 0;
          // for (auto it : patch_w)
          //   {
          //     Vec<3> vi_minus_vj = vertex->xy - it.first->xy;
          //     Vec<3> Tij = vi_minus_vj - VecTransVecMatrix<3>(normal) * (vi_minus_vj);
          //     Tij *= 1./L2Norm(Tij);
          //     double kij = 2 * InnerProduct(normal,vi_minus_vj)/L2Norm(vi_minus_vj);
          //     M += it.second * kij * VecTransVecMatrix<3>(Tij);
          //   }
          // double area = 0;
          // for (auto trig : vertex->neighbours)
          //   area += trig->area;
          // //double energy = 0;
          // auto w = linalg::dsyevc3(M);
          // return py::make_tuple(w[0],w[1],w[2]);
        });

#ifdef TEST_GEOMETRY

  py::class_<TestGeometry, shared_ptr<TestGeometry>, VoxelSTLGeometry>
    (m,"TestGeometry")
    .def(py::init([](py::array_t<double> verts, py::array_t<size_t> trigs,
                     shared_ptr<spdlog::logger> log)
                  {
                    Array<double> averts(verts.size(), (double*) verts.data());
                    Array<size_t> atrigs(trigs.size(), (size_t*) trigs.data());
                    return TestGeometry(averts,atrigs,log);
                  }), py::arg("vertices"), py::arg("triangles"),
         py::arg("logger")=CreateLogger("TestGeometry"))
    .def("GetBoundingBox ",[](TestGeometry& self)
                          {
                            return py::make_tuple(0,1,0,1,0,1);
                          })
    ;
#endif //TEST_GEOMETRY
}


PYBIND11_MODULE(libvoxel2stl,m)
{
  py::module::import("pyspdlog");
  ExportVoxel2STL(m);
}
