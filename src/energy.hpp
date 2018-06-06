#ifndef __FILE_V2S_ENERGY_HPP
#define __FILE_V2S_ENERGY_HPP

#include "linalg.hpp"

namespace voxel2stl
{

  template<typename E>
  class PatchEnergy
  {
  public:
    inline static double Energy(VoxelSTLGeometry::Vertex* vertex)
    {
      double e = 0;
      for(auto v : vertex->patch)
        e += E::Energy(v);
      return e;
    }
  };

  class FirstEnergy
  {
  public:
    inline static double Energy(VoxelSTLGeometry::Vertex* vertex)
    {
      double E = 0;
      for(auto dof : vertex->patch)
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
                  E += fair * cosAngle/sqrt(trig->area*trig->area + trig2->area*trig2->area);
                }
            }
        }
      // double area = 0.;
      // for (auto trig : vertex->neighbours)
      //   area += trig->area;
      // E *= sqrt(area);

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

  // from gauss-bonnet style curvature
  class AngleDeficit
  {
  public:
    inline static double Energy(VoxelSTLGeometry::Vertex* vertex)
    {
      double energy = 2*M_PI;
      double area = 0;
      for(auto tri : vertex->neighbours)
        {
          auto v1 = tri->OtherVertices(vertex,0);
          auto v2 = tri->OtherVertices(vertex,1);
          area += tri->area;
          double a = L2Norm(v1->xy - vertex->xy);
          double b = L2Norm(v2->xy - vertex->xy);
          double c = L2Norm(v1->xy - v2->xy);
          double alpha = acos((a*a+b*b-c*c)/(2*a*b));
          energy += alpha;
        }
      energy *= 3./area;
      return energy;
    }
  };

  template<int H>
  class VecTransVecMatrix : public MatExpr<VecTransVecMatrix<H>>
  {
  private:
    Vec<H>& vec;
  public:
    VecTransVecMatrix(Vec<H>& avec) : vec(avec) { }
    double operator() (int i) const
    {
      return vec[i%H]*vec[i/H];
    }
    ///
    double operator() (int i, int j) const { return vec[i]*vec[j]; }

    /// the height
    int Height () const { return H; }
    /// the width
    int Width () const { return H; }
  };

  class TaubinIntegralFormulation
  {
  public:
    inline static double Energy(VoxelSTLGeometry::Vertex* vertex)
    {
      std::map<VoxelSTLGeometry::Vertex*,double> patch_w;
      Vec<3> normal = 0;
      for(auto trig : vertex->neighbours)
        {
          normal += trig->area * trig->n;
          patch_w[trig->OtherVertices(vertex,0)] += trig->area;
          patch_w[trig->OtherVertices(vertex,1)] += trig->area;
          patch_w[trig->OtherVertices(vertex,0)] = 1;
          patch_w[trig->OtherVertices(vertex,1)] = 1;
        }
      normal /= L2Norm(normal);
      double normw = 0;
      for (auto it : patch_w)
        normw += it.second*it.second;
      double fac = 1./sqrt(normw);
      for(auto it : patch_w)
        patch_w[it.first] *= fac;
      Mat<3> M;
      M = 0;
      for (auto it : patch_w)
        {
          Vec<3> vi_minus_vj = vertex->xy - it.first->xy;
          Vec<3> Tij = vi_minus_vj - VecTransVecMatrix<3>(normal) * (vi_minus_vj);
          Tij *= 1./L2Norm(Tij);
          double kij = 2 * InnerProduct(normal,vi_minus_vj)/L2Norm(vi_minus_vj);
          M += it.second * kij * VecTransVecMatrix<3>(Tij);
        }
      double area = 0;
      for (auto trig : vertex->neighbours)
        area += trig->area;
      //double energy = 0;
      auto w = linalg::dsyevc3(M);
      //cout << "evals = " << w[0] << ", " << w[1] << ", " << w[2] << endl;
      return L2Norm(w); // /sqrt(area); //(fabs(w[0]) + fabs(w[1]) + fabs(w[2]))/area;

      // energy += (M(1,1)*M(2,2)-M(2,1)*M(1,2))*(M(1,1)*M(2,2)-M(2,1)*M(1,2));
      // energy += (-M(1,0)*M(2,2)+M(2,0)*M(1,2))*(-M(1,0)*M(2,2)+M(2,0)*M(1,2));
      // energy += (M(1,0)*M(2,1)-M(2,0)*M(1,1))*(M(1,0)*M(2,1)-M(2,0)*M(1,1));

      // energy += (-M(0,1)*M(2,2)+M(2,1)*M(0,2))*(-M(0,1)*M(2,2)+M(2,1)*M(0,2));
      // energy += (M(0,0)*M(2,2)-M(2,0)*M(0,2))*(M(0,0)*M(2,2)-M(2,0)*M(0,2));
      // energy += (-M(0,0)*M(2,1)+M(2,0)*M(0,1))*(-M(0,0)*M(2,1)+M(2,0)*M(0,1));

      // energy += (M(0,1)*M(1,2)-M(1,1)*M(0,2))*(M(0,1)*M(1,2)-M(1,1)*M(0,2));
      // energy += (-M(0,0)*M(1,2)+M(1,0)*M(0,2))*(-M(0,0)*M(1,2)+M(1,0)*M(0,2));
      // energy += (M(0,0)*M(1,1)-M(1,0)*M(0,1))*(M(0,0)*M(1,1)-M(1,0)*M(0,1));
      // return sqrt(energy)/area;
    }
  };

  template<typename E1, typename E2, int W>
  class WeightedEnergy
  {
  public:
    inline static double Energy(VoxelSTLGeometry::Vertex* vertex)
    {
      double fac = double(W)/100.;
      return fac *E1::Energy(vertex) + (1.-fac)*E2::Energy(vertex);
    }
  };
}


# endif // __FILE_V2S_ENERGY_HPP
