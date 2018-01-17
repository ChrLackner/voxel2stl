
#include "voxel2stl.hpp"
#include <omp.h>

namespace voxel2stl
{
  template<typename E>
  void NewtonSmoother<E> :: Apply()
  {
    int num_threads;
#pragma omp parallel
    {
      num_threads = omp_get_num_threads();
    }
    double e = 0; double e_diff = 0;
    log->info("Minimize energy with " + std::to_string(num_threads) + " threads");
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
  }

  template class NewtonSmoother<SimpleEnergy>;
  template class NewtonSmoother<FirstEnergy>;
}
