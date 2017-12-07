
#include "voxel2stl.hpp"
#include <omp.h>
#include <set>

namespace voxel2stl
{

  double Smoother :: Newton(std::function<double (double)>  func,
                            double r0, double &energydifference, double & energy){
	double r=r0;
	energy = func(r0);
	//newton method
	size_t count = 0;
	double alpha = 1.0;
	size_t countalpha = 1;
	double eps = 1e-2; //try 1e-2 or 1e-3
	double h = 1e-4; //h must be smaller than eps
	double w=1;
	while (count<20 && r>eps && r<1-eps && abs(w)>eps){
		double fp = func(r+h);
		double fm = func(r-h);
		double f = func(r);
		w = h/2* (fp-fm) / (fp-2*f+fm);
		// turn w if it is wrong orientated
		if (w*(fp-fm) < 0) {w = (-1)*w;}
		alpha = 1.0;
		countalpha = 1;
		while(countalpha < 5 && (r-alpha*w<=eps || r-alpha*w>=1-eps || func(r-alpha*w)>=f)){
			alpha = alpha/3;
			countalpha += 1;
		}
		if (countalpha < 5 /*&& r-alpha*w>eps && r-alpha*w<1-eps*/){
			r = r-alpha*w;
		}
		else{
			count = 100;
		}
		count += 1;
	}
	energydifference += (energy - func(/*0.5*(*/ r /*+r0)*/));
	return r;
  }

  void FirstSmoother :: Apply()
  {
    int num_threads;
#pragma omp parallel
    {
      num_threads = omp_get_num_threads();
    }
    double e = 0; double e_diff = 0;
    log->info("Minimize energy with " + std::to_string(num_threads) + " threads");
    for (size_t cluster = 0; cluster<geo->vertex_clustering.Size(); cluster++)
      {
#pragma omp parallel for
        for (size_t i = 0; i < geo->vertex_clustering[cluster]->Size();i++)
          {
            auto vertex = (*geo->vertex_clustering[cluster])[i];
            double energy_difference = 0;
            double energy = 0;
            energy = 0;
            energy_difference = 0;
            /*energie =
             * fläche der nachbardreiecke
             * abweichungen der normalen im dreieck
             * abweichung der normalen zu nachbardreiecken
             * also: cos(winkel)
             */
            Vec<3,double> x = vertex->x;
            Vec<3,double> v = (vertex->y)-(vertex->x);
            int nn = vertex->nn;
            double r = vertex->ratio;
            //Lambda function in Newton
            std::set<VoxelSTLGeometry::Vertex*> dofs;
            std::set<VoxelSTLGeometry::Vertex*>::iterator dof_it;
            std::set<VoxelSTLGeometry::Triangle*> tris;
            std::set<VoxelSTLGeometry::Triangle*>::iterator tri_it;
            dofs.insert(vertex);
            for (int j=0; j<nn; j++){
              tris.insert(vertex->neighbours[j]);
            }
            for (tri_it = tris.begin(); tri_it != tris.end(); tri_it++){
              for (int k=0; k<2; k++){
                dofs.insert((*tri_it)->OtherVertices(vertex,k));
              }
            }
            r = Newton([&](double r){
                double E = 0;
                VoxelSTLGeometry::Triangle* me;
                VoxelSTLGeometry::Triangle* he;
                VoxelSTLGeometry::Vertex* me_v1;
                VoxelSTLGeometry::Vertex* me_v2;
                int fair = 0;
                double cosAngle = 0;
                vertex->SetRatio(r);
                for (dof_it = dofs.begin(); dof_it != dofs.end(); dof_it++){
                  for(int j=0; j<(*dof_it)->nn; j++){
                    me = (*dof_it)->neighbours[j];
                    me_v1 = me->OtherVertices((*dof_it),1);
                    me_v2 = me->OtherVertices((*dof_it),2);
                    for(int k=j+1; k<(*dof_it)->nn; k++){
                      he = (*dof_it)->neighbours[k];
                      if(he->doIhave(me_v1) || he->doIhave(me_v2)){
                        fair = 1;
                      }else{
                        fair = 2;
                      }
                      cosAngle = InnerProduct(me->n,he->n);
                      cosAngle = (1-cosAngle)*(1-cosAngle); //von -1,1 nach 0,2 und ^2
                      E += fair*cosAngle/ sqrt(me->area+he->area); //penalty durch area
                    }
                  }
                }
                for (int j = 0; j < nn; j++){
                  E += weight_area*vertex->neighbours[j]->area;
                }
                return E;}, r, energy_difference, energy);
#pragma omp critical
            {
              e += energy;
              e_diff += energy_difference;
            }

            vertex->SetRatio((1-weight_minimum)*vertex->ratio+(weight_minimum)*r);
            for (int j = 0; j < nn; j++){
              vertex->neighbours[j]->UpdateNormal();
            }
          }
      }
  }

}
