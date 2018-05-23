# ifndef __FILE_SMOOTHER_HPP
# define __FILE_SMOOTHER_HPP

#include <omp.h>
#include <math.h>

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
    virtual Array<float> CalcEnergy() const = 0;
  };

  template<typename E>
  class NewtonSmoother : public Smoother
  {
  public:
    NewtonSmoother(shared_ptr<VoxelSTLGeometry> geo, shared_ptr<spdlog::logger> log)
      : Smoother(geo,log) { ; }

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
            {
              double energy, energydifference;
              std::tie(energy,energydifference) = Newton((*vertex_clustering[cluster])[i]);
              #pragma omp critical (e)
              {
                e += energy;
                e_diff += energydifference;
              }
              // vertex->SetRatio((1-weight_minimum)*vertex->ratio+(weight_minimum)*r);
            }
        }
      log->info("Smoothing done.");
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
          SPDLOG_DEBUG(log,"Newton step " + std::to_string(count));
          double f = E::Energy(vertex);
          SPDLOG_DEBUG(log,"E::Energy = " + std::to_string(f));
          vertex->SetRatio(vertex->ratio+h);
          double fp = E::Energy(vertex);
          vertex->SetRatio(vertex->ratio-2*h);
          double fm = E::Energy(vertex);
          vertex->SetRatio(vertex->ratio + h);
          if(abs((fp-fm)/(2*h*f))<1e-10)
            break;
          // w = h/2* (fp-fm) / (fp-2*f+fm);
          w = f / ((fp-fm) / (2*h));
          // cout << "w = " << w << endl;
          double alpha = 1.0;
          while (vertex->ratio-alpha*w<eps || vertex->ratio-alpha*w>1-eps)
            alpha *= 0.9;
          // cout << "start with alpha = " << alpha << endl;
          while(abs(alpha * w) > 1e-8)
            {
              vertex->SetRatio(vertex->ratio - alpha * w);
              if(E::Energy(vertex) <= f)
                break;
              vertex->ratio += alpha * w;
              // cout << "E::Energy not reduced, Energy now = " << Energy(vertex) << endl;
              alpha *= 0.8;
            }
          if (abs(alpha*w) < 9e-7)
              break;
          SPDLOG_DEBUG(log,"Energy reduced to " + std::to_string(E::Energy(vertex)));
          count += 1;
        }
      energydifference += (energy - E::Energy(vertex));
      return std::make_tuple(energy,energydifference);
    }
  };

  class FirstEnergy
  {
  public:
    inline static double Energy(VoxelSTLGeometry::Vertex* vertex)
    {
      std::set<VoxelSTLGeometry::Vertex*> dofs;
      dofs.insert(vertex);
      for (auto tri : vertex->neighbours)
        for(int k=0; k<2; k++)
          dofs.insert(tri->OtherVertices(vertex,k));

      double E = 0;
      for(auto dof : dofs)
        {
          for(int j = 0; j<dof->nn; j++)
            {
              auto trig = dof->neighbours[j];
              auto trig_v1 = trig->OtherVertices(dof,1);
              auto trig_v2 = trig->OtherVertices(dof,2);
              for (int k = j+1; k<dof->nn; k++)
                {
                  int fair;
                  auto trig2 = dof->neighbours[k];
                  if(trig2->doIhave(trig_v1) || trig2->doIhave(trig_v2))
                    fair = 1;
                  else
                    fair = 1;
                  double cosAngle = InnerProduct(trig->n, trig2->n);
                  cosAngle = (1-cosAngle)*(1-cosAngle);
                  E += fair * cosAngle; ///sqrt(trig->area*trig->area + trig2->area*trig2->area);
                }
            }
        }
      // double area = 0.;
      // for (auto trig : vertex->neighbours)
      //   area += trig->area;
      // E *= weight_area * area;

      return E;
    }
  };

  class SimpleEnergy
  {
  public:
    inline static double Energy(VoxelSTLGeometry::Vertex* vertex)
    {
      std::set<VoxelSTLGeometry::Vertex*> dofs;
      for (auto tri : vertex->neighbours)
        for(int k=0; k<2; k++)
          dofs.insert(tri->OtherVertices(vertex,k));
      Vec<3> sum = 0.0;
      for(auto dof : dofs)
        sum += dof->xy;
      sum *= 1./(dofs.size());
      return abs(L2Norm(sum-vertex->xy));
    }
  };

  // from http://www.cc.ac.cn/06research_report/0601.pdf
  class AngleDeficit
  {
  public:
    inline static double Energy(VoxelSTLGeometry::Vertex* vertex)
    {
      double energy = 2*M_PI;
      double area = 0;
      for(auto tri : vertex->neighbours)
        {
          area += tri->area;
          double a = L2Norm(tri->OtherVertices(vertex,1)->xy - vertex->xy);
          double b = L2Norm(tri->OtherVertices(vertex,2)->xy - vertex->xy);
          double c = L2Norm(tri->OtherVertices(vertex,1)->xy - tri->OtherVertices(vertex,2)->xy);
          double alpha = acos((a*a+b*b-c*c)/(2*a*b));
          energy -= alpha;
        }
      energy *= 3./area;
      return energy;
    }
  };

}

# endif // __FILE_SMOOTHER_HPP
