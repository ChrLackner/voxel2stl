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
      e *= 1./(vertex->patch.size());
      e += E::Energy(vertex);
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
                    fair = 3;
                  double cosAngle = InnerProduct(trig->n, trig2->n);
                  cosAngle = (1-cosAngle)*(1-cosAngle);
                  E += fair * cosAngle/sqrt(trig->area*trig->area + trig2->area*trig2->area);
                }
            }
        }
      double area = 0.;
      for (auto trig : vertex->neighbours)
        area += trig->area;
      E *= sqrt(area);

      // scaling for energy
      return E/100;
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
    Vec<H> vec;
  public:
    VecTransVecMatrix(const Vec<H>& avec) : vec(avec) { }
    double operator() (int i) const
    {
      return vec[i%H]*vec[i/H];
    }
    double operator() (int i, int j) const { return vec[i]*vec[j]; }

    int Height () const { return H; }
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
        }
      normal /= L2Norm(normal);
      double normw = 0;
      for (auto it : patch_w)
        normw += it.second;
      double fac = 1./normw;
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

      Vec<3> e1 = { 1.,0.,0.};
      Vec<3> w;
      if(L2Norm(e1-normal) > L2Norm(e1+normal))
        w = 1./L2Norm(e1-normal) * (e1-normal);
      else
        w = 1./L2Norm(e1+normal) * (e1 + normal);
      Mat<3> Q = Id<3>() - 2 * VecTransVecMatrix<3>(w);
      // first row and column of Q is orthogonal to normal -> first row and column of T is 0
      Mat<3> T = Trans(Q) * M * Q;
      double trace = T(1,1) + T(2,2);
      double det = T(1,1)*T(2,2) - T(1,2)*T(2,1);
      // eigs are trace/2 +- sqrt(trace**2/4-det)
      double eig1 = trace/2+sqrt(trace*trace/4-det);
      double eig2 = trace/2 - sqrt(trace*trace/4-det);
      // principal curvature is:
      double kp1 = 3 * eig1 - eig2;
      double kp2 = 3 * eig2 - eig1;
      double min_area = 1e99;
      for(auto trig : vertex->neighbours)
        min_area = min2(min_area, trig->area);
      return sqrt(kp1*kp1 + kp2*kp2);
    }
  };

  template<typename E1, typename E2, int W>
  class WeightedEnergy
  {
  public:
    inline static double Energy(VoxelSTLGeometry::Vertex* vertex)
    {
      double fac = double(W)/1000.;
      return fac *E1::Energy(vertex) + (1.-fac)*E2::Energy(vertex);
    }
  };

  // from: https://arxiv.org/pdf/cs/0510090.pdf
  class CurvatureEstimation
  {
  public:
    inline static double Energy(VoxelSTLGeometry::Vertex* vert)
    {
      auto vals = EigenValues(vert);
      double area = 0.;
      for(auto trig : vert->neighbours)
        area += trig->area;
      return L2Norm(vals) * sqrt(area);
      // auto grad = vertexGradient(vert);
      // auto normal = vertexNormal(vert);
      // Mat<3> basis = Identity(3);
      // for(auto i : Range(3))
      //   basis.Col(i) -= InnerProduct(basis.Col(i), normal) * normal;
      // int smallest_index = 0;
      // double smallest_value = 1e99;
      // for(auto i : Range(3))
      //   {
      //     double val = L2Norm(basis.Col(i));
      //     if (val < smallest_value)
      //       {
      //         smallest_value = val;
      //         smallest_index = i;
      //       }
      //   }
      // std::vector<int> indices;
      // for(int i : Range(3))
      //   if(i != smallest_index)
      //     indices.push_back(i);
      // Vec<3> t1 = basis.Col(indices[0]);
      // Vec<3> t2 = basis.Col(indices[1]);
      // t1 *= 1./L2Norm(t1);
      // t2 *= 1./L2Norm(t2);
      // t2 -= InnerProduct(t1,t2) * t1;
      // t2 *= 1./L2Norm(t2);
      // t1 -= InnerProduct(normal, t1) * normal;
      // t2 -= InnerProduct(normal, t2) * normal;
      // t1 *= 1./L2Norm(t1);
      // t2 *= 1./L2Norm(t2);
      // t2 -= InnerProduct(t1,t2) * t1;
      // t2 *= 1./L2Norm(t2);
      // // cout << "***************************************************************" << endl;
      // // cout << "normal = " << normal << endl;
      // // cout << "indices = " << indices.size() << ": " << indices[0] << ", " << indices[1] << endl;
      // // cout << "grad = " << grad << endl;
      // // cout << "t1 = " << t1 << endl;
      // // cout << "t2 = " << t2 << endl;
      // Mat<2> mat;
      // mat(0,0) = InnerProduct(t1,grad*t1);
      // mat(0,1) = InnerProduct(t1,grad*t2);
      // mat(1,0) = InnerProduct(t2,grad*t1);
      // mat(1,1) = InnerProduct(t2,grad*t2);
      // // cout << "mat = " << mat << endl;
      // auto curv = Det(mat);
      // // cout << "curvature = " << curv << endl;
      // return fabs(curv);
    }

    inline static Vec<3> EigenValues(VoxelSTLGeometry::Vertex* vert)
    {
      auto grad = vertexGradient(vert);
      auto w = linalg::dsyevc3(grad);
      return w;
    }

    inline static Vec<3> vertexNormal(VoxelSTLGeometry::Vertex* vert)
    {
      auto weights = computePatchWeights(vert);
      Vec<3> normal = 0.;
      for(auto i : Range(vert->nn))
        normal += weights[i] * vert->neighbours[i]->n;
      normal *= 1./L2Norm(normal);
      return normal;
    }

    inline static std::vector<double> computePatchWeights(VoxelSTLGeometry::Vertex* vert)
    {
      std::vector<double> weights;
      weights.reserve(vert->nn);
      for(auto trig : vert->neighbours)
        weights.push_back(1./L2Norm2(centroid(trig)-vert->xy));
      double sum = 0.;
      for(auto val : weights)
        sum += val;
      for(auto& val : weights)
        val /= sum;
      return weights;
    }

    inline static Vec<3> centroid(VoxelSTLGeometry::Triangle* trig)
    {
      return 1./3 * (trig->v1->xy + trig->v2->xy + trig->v3->xy);
    }

    inline static Mat<3,3> faceGradient(VoxelSTLGeometry::Vertex* vert, VoxelSTLGeometry::Triangle* trig)
    {
      Mat<2> mat;
      Mat<2> invmat;
      auto vj = trig->OtherVertices(vert,0);
      auto vk = trig->OtherVertices(vert,1);
      mat(0,0) = InnerProduct(vj->xy - vert->xy, vj->xy - vert->xy);
      mat(0,1) = InnerProduct(vj->xy - vert->xy, vk->xy - vert->xy);
      mat(1,0) = InnerProduct(vj->xy - vert->xy, vk->xy - vert->xy);
      mat(1,1) = InnerProduct(vk->xy - vert->xy, vk->xy - vert->xy);
      CalcInverse(mat,invmat);
      Mat<3> result;
      Vec<3> gvj = vertexNormal(vj);
      Vec<3> gvk = vertexNormal(vk);
      Vec<3> gvert = vertexNormal(vert);
      for(auto i : Range(3))
        {
          Vec<2> gvals = { gvj[i] - gvert[i] , gvk[i] - gvert[i] };
          Vec<2> sol = invmat * gvals;
          result.Col(i) = sol[0] * (vj->xy - vert->xy) + sol[1] * (vk->xy - vert->xy);
        }
      return result;
    }

    inline static Mat<3> vertexGradient(VoxelSTLGeometry::Vertex* vert)
    {
      Mat<3> grad = 0.;
      auto weights = computePatchWeights(vert);
      for(auto i : Range(vert->nn))
        grad += weights[i] * faceGradient(vert, vert->neighbours[i]);
      return grad;
    }
  };

}

# endif // __FILE_V2S_ENERGY_HPP
