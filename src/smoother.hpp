# ifndef __FILE_SMOOTHER_HPP
# define __FILE_SMOOTHER_HPP

#include <omp.h>
#include <math.h>

namespace voxel2stl
{
  // Constants
#define M_SQRT3    1.73205080756887729352744634151   // sqrt(3)

// Macros
#define SQR(x)      ((x)*(x))                        // x^2
// ----------------------------------------------------------------------------
  inline Vec<3> dsyevc3(const Mat<3>& A)
// ----------------------------------------------------------------------------
// Calculates the eigenvalues of a symmetric 3x3 matrix A using Cardano's
// analytical algorithm.
// Only the diagonal and upper triangular parts of A are accessed. The access
// is read-only.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The symmetric input matrix
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
{
  double m, c1, c0;

  // Determine coefficients of characteristic poynomial. We write
  //       | a   d   f  |
  //  A =  | d*  b   e  |
  //       | f*  e*  c  |
  double de = A(0,1) * A(1,2);                                    // d * e
  double dd = SQR(A(0,1));                                         // d^2
  double ee = SQR(A(1,2));                                         // e^2
  double ff = SQR(A(0,2));                                         // f^2
  m  = A(0,0) + A(1,1) + A(2,2);
  c1 = (A(0,0)*A(1,1) + A(0,0)*A(2,2) + A(1,1)*A(2,2))        // a*b + a*c + b*c - d^2 - e^2 - f^2
          - (dd + ee + ff);
  c0 = A(2,2)*dd + A(0,0)*ee + A(1,1)*ff - A(0,0)*A(1,1)*A(2,2)
            - 2.0 * A(0,2)*de;                                     // c*d^2 + a*e^2 + b*f^2 - a*b*c - 2*f*d*e)

  double p, sqrt_p, q, c, s, phi;
  p = SQR(m) - 3.0*c1;
  q = m*(p - (3.0/2.0)*c1) - (27.0/2.0)*c0;
  sqrt_p = sqrt(fabs(p));

  phi = 27.0 * ( 0.25*SQR(c1)*(p - c1) + c0*(q + 27.0/4.0*c0));
  phi = (1.0/3.0) * atan2(sqrt(fabs(phi)), q);

  c = sqrt_p*cos(phi);
  s = (1.0/M_SQRT3)*sqrt_p*sin(phi);

  Vec<3> w;
  w[1]  = (1.0/3.0)*(m - c);
  w[2]  = w[1] + s;
  w[0]  = w[1] + c;
  w[1] -= s;
  return w;
}

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
          // double f = E::Energy(vertex);
          // vertex->SetRatio(vertex->ratio + h);
          // double fp = E::Energy(vertex);
          // vertex->SetRatio(vertex->ratio + h);
          // double f2p = E::Energy(vertex);
          // vertex->SetRatio(vertex->ratio - 4*h);
          // double f2m = E::Energy(vertex);
          // vertex->SetRatio(vertex->ratio + h);
          // double fm = E::Energy(vertex);
          // vertex->SetRatio(vertex->ratio + h);
          // double df = (8*fp-8*fm-f2p+f2m)/(12*h);
          // double alpha = 1.;
          // while(vertex->ratio-alpha*w < eps || vertex->ratio-alpha*w>1-eps)
          //   alpha *= 0.9;
          // double startratio = vertex->ratio;
          // while(abs(alpha*w) > 1e-8)
          //   {
          //     vertex->SetRatio(startratio-alpha*w);
          //     if(E::Energy(vertex) <= f)
          //       break;
          //     alpha *= 0.9;
          //   }
          // count += 1;
          SPDLOG_DEBUG(log,"Newton step " + std::to_string(count));
          double f = E::Energy(vertex);
          SPDLOG_DEBUG(log,"E::Energy = " + std::to_string(f));
          vertex->SetRatio(vertex->ratio+h);
          double fp = E::Energy(vertex);
          vertex->SetRatio(vertex->ratio-2*h);
          double fm = E::Energy(vertex);
          vertex->SetRatio(vertex->ratio + h);
          // if(abs((fp-fm)/(2*h*f))<1e-10)
          //   break;
          w = h/2* (fp-fm) / (fp-2*f+fm);
          if(w*(fp-fm)<0) w *= -1;
          // w = f / ((fp-fm) / (2*h));
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
                    fair = 2;
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
      return L2Norm(sum-vertex->xy);
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
          normal += /*trig->area */ trig->n;
          // patch_w[trig->OtherVertices(vertex,0)] += trig->area;
          // patch_w[trig->OtherVertices(vertex,1)] += trig->area;
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
      double energy = 0;
      auto w = dsyevc3(M);
      // cout << "evals = " << w[0] << ", " << w[1] << ", " << w[2] << endl;
      return fabs(w[0]) + fabs(w[1]) + fabs(w[2]);
      // energy += (M(1,1)*M(2,2)-M(2,1)*M(1,2))*(M(1,1)*M(2,2)-M(2,1)*M(1,2));
      // energy += (-M(1,0)*M(2,2)+M(2,0)*M(1,2))*(-M(1,0)*M(2,2)+M(2,0)*M(1,2));
      // energy += (M(1,0)*M(2,1)-M(2,0)*M(1,1))*(M(1,0)*M(2,1)-M(2,0)*M(1,1));

      // energy += (-M(0,1)*M(2,2)+M(2,1)*M(0,2))*(-M(0,1)*M(2,2)+M(2,1)*M(0,2));
      // energy += (M(0,0)*M(2,2)-M(2,0)*M(0,2))*(M(0,0)*M(2,2)-M(2,0)*M(0,2));
      // energy += (-M(0,0)*M(2,1)+M(2,0)*M(0,1))*(-M(0,0)*M(2,1)+M(2,0)*M(0,1));

      // energy += (M(0,1)*M(1,2)-M(1,1)*M(0,2))*(M(0,1)*M(1,2)-M(1,1)*M(0,2));
      // energy += (-M(0,0)*M(1,2)+M(1,0)*M(0,2))*(-M(0,0)*M(1,2)+M(1,0)*M(0,2));
      // energy += (M(0,0)*M(1,1)-M(1,0)*M(0,1))*(M(0,0)*M(1,1)-M(1,0)*M(0,1));
      // return sqrt(energy);
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

# endif // __FILE_SMOOTHER_HPP
