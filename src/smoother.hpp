# ifndef __FILE_SMOOTHER_HPP
# define __FILE_SMOOTHER_HPP

#include <omp.h>
#include "energy.hpp"

namespace voxel2stl
{

  class Smoother : public pyspdlog::LoggedClass
  {
  protected:
    shared_ptr<VoxelSTLGeometry> geo;
  public:
    Smoother(shared_ptr<VoxelSTLGeometry> _geo, shared_ptr<spdlog::logger> log)
      : geo(_geo), pyspdlog::LoggedClass(log) { ; }

    virtual void Apply() = 0;
    virtual Array<float> CalcEnergy() const
    {
      cout << "CalcEnergy not implemented for this type of Smoother!" << endl;
      return Array<float>();
    }
    virtual double CalcVertexEnergy(size_t index) const
    {
      cout << "CalcVertexEnergy not implemented for this Energy!" << endl;
      return 0;
    }
  };

  template<typename E>
  class NewtonSmoother : public Smoother
  {
  public:
    NewtonSmoother(shared_ptr<VoxelSTLGeometry> geo, shared_ptr<spdlog::logger> log)
      : Smoother(geo,log)
    {
      // geo->CreateClusters<NewtonClass>();
    }

    virtual void Apply() override
    {
      int num_threads;
#pragma omp parallel
      {
        num_threads = omp_get_num_threads();
      }
      double e = 0; double e_diff = 0;
      log->info("Minimize energy with " + std::to_string(num_threads) + " threads...");
      auto& vertex_clustering = geo->GetVertexClustering();
      for (size_t cluster = 0; cluster<vertex_clustering.Size(); cluster++)
        {
          #pragma omp parallel for
          for (size_t i = 0; i < vertex_clustering[cluster]->Size();i++)
            Newton((*vertex_clustering[cluster])[i]);
        }
      log->info("done.");
    }
    virtual Array<float> CalcEnergy() const override
    {
      Array<float> energy;
      energy.SetAllocSize(geo->GetNTriangles() * 3);
      for (auto i : Range(geo->GetNTriangles()))
        for(auto v : { geo->GetTriangle(i).v1,
                       geo->GetTriangle(i).v2,
                       geo->GetTriangle(i).v3})
          energy.Append(E::Energy(v));
      return std::move(energy);
    }

    virtual double CalcVertexEnergy(size_t index) const override
    {
      return E::Energy(geo->GetVertex(index));
    }

  protected:
    inline std::tuple<double,double> Newton(VoxelSTLGeometry::Vertex* vertex)
    {
      double energy = E::Energy(vertex);
      double energydifference = 0;
      //newton method
      int count = 0;
      double eps = 1e-3; //try 1e-2 or 1e-3
      double h = 1e-6; //h must be smaller than eps
      double w=1;
      // static int totcount = 0;
      // if(totcount > 4)
      //   count = 100;
      // totcount++;
      while (count<20 && vertex->ratio>eps && vertex->ratio < 1-eps && abs(w)>1e-10)
        {
          double f = E::Energy(vertex);
          vertex->SetRatio(vertex->ratio + h);
          double fp = E::Energy(vertex);
          vertex->SetRatio(vertex->ratio + h);
          double f2p = E::Energy(vertex);
          vertex->SetRatio(vertex->ratio - 4*h);
          double f2m = E::Energy(vertex);
          vertex->SetRatio(vertex->ratio + h);
          double fm = E::Energy(vertex);
          vertex->SetRatio(vertex->ratio + h);
          double df = (8*fp-8*fm-f2p+f2m)/(12*h);
          double alpha = 1.;
          while(vertex->ratio-alpha*df < eps || vertex->ratio-alpha*df>1-eps)
            alpha *= 0.9;
          double startratio = vertex->ratio;
          while(abs(alpha*df) > 1e-8)
            {
              vertex->SetRatio(startratio-alpha*df);
              if(E::Energy(vertex) <= f)
                break;
              alpha *= 0.9;
            }
          count += 1;

          // **********************************************************

          // SPDLOG_DEBUG(log,"Newton step " + std::to_string(count));
          // double f = E::Energy(vertex);
          // SPDLOG_DEBUG(log,"E::Energy = " + std::to_string(f));
          // vertex->SetRatio(vertex->ratio+h);
          // double fp = E::Energy(vertex);
          // vertex->SetRatio(vertex->ratio-2*h);
          // double fm = E::Energy(vertex);
          // vertex->SetRatio(vertex->ratio + h);
          // // if(abs((fp-fm)/(2*h*f))<1e-10)
          // //   break;
          // w = h/2* (fp-fm) / (fp-2*f+fm);
          // if(w*(fp-fm)<0) w *= -1;
          // // w = f / ((fp-fm) / (2*h));
          // // cout << "w = " << w << endl;
          // double alpha = 1.0;
          // while (vertex->ratio-alpha*w<eps || vertex->ratio-alpha*w>1-eps)
          //   alpha *= 0.9;
          // // cout << "start with alpha = " << alpha << endl;
          // while(abs(alpha * w) > 1e-8)
          //   {
          //     vertex->SetRatio(vertex->ratio - alpha * w);
          //     if(E::Energy(vertex) <= f)
          //       break;
          //     vertex->ratio += alpha * w;
          //     // cout << "E::Energy not reduced, Energy now = " << Energy(vertex) << endl;
          //     alpha *= 0.8;
          //   }
          // if (abs(alpha*w) < 9e-7)
          //     break;
          // SPDLOG_DEBUG(log,"Energy reduced to " + std::to_string(E::Energy(vertex)));
          // count += 1;
        }
      energydifference += (energy - E::Energy(vertex));
      return std::make_tuple(energy,energydifference);
    }
  };

}

# endif // __FILE_SMOOTHER_HPP
