
#include "voxel2stl.hpp"
#include <omp.h>
#include <set>


namespace voxel2stl
{
  VoxelSTLGeometry::VoxelSTLGeometry(shared_ptr<VoxelData> adata,
                                     shared_ptr<Array<int>> amaterials,
                                     shared_ptr<Array<int>> aboundaries,
                                     shared_ptr<spdlog::logger> log)
    : pyspdlog::LoggedClass(log),
      data(adata), materials(amaterials), boundaries(aboundaries)
  {
    m = data->Getm();
#pragma omp parallel
    {
      num_threads = omp_get_num_threads();
    }
    GenerateTrianglesAndVertices();
    PartitionVertices();
  }

  void VoxelSTLGeometry :: ApplySmoothingStep(bool subdivision, double weight_area,
                                              double weight_minimum)
  {
    if (subdivision)
      {
        SubdivisionTriangles();
        PartitionVertices();
      }
    MinimizeEnergy(weight_area,weight_minimum);
  }

  void VoxelSTLGeometry :: KeepInside (shared_ptr<VoxelSTLGeometry> other)
  {
    auto nvert = other->GetVertices().Size();
#pragma omp parallel for
    for (size_t i = 0; i < nvert; i++)
      {
        Vertex *vother = &*(other->GetVertices()[i]);
        Vertex *closest = nullptr;
        FindClosest(vother, &closest);
        
      }
  }

  void VoxelSTLGeometry :: WriteSTL(string filename)
  {
    log->trace("Start writing to stl file");
    ofstream stlout;
    stlout.open(filename);
    stlout.precision(12);
    stlout << "solid" << endl;
    Vec<3,double> p1;
    Vec<3,double> p2;
    Vec<3,double> p3;
    Vec<3,double> n;
    for (auto i : Range(triangles.Size())){
      p1 = triangles[i]->v1->xy;
      p2 = triangles[i]->v2->xy;
      p3 = triangles[i]->v3->xy;
      n = triangles[i]->n;
      if (L2Norm(n) < 1e-10 || L2Norm(p1 - p2) < 1e-10 || L2Norm(p3 - p2) < 1e-10)
        {
          log->error("Error in triangle!");
          return;
	}
      Vec<3,double> normal;
      Vec<3,double> temp1 = p2-p1;
      Vec<3,double> temp2 = p3-p1;
      normal = Cross(temp1,temp2);

      if (InnerProduct(normal, n) < 0)
	{
          Vec<3,double> temp;
          temp = p2;
          p2 = p3;
          p3 = temp;
	}

      stlout << "facet normal " << n[0] << " " << n[1] << " " << n[2] << "\n";
      stlout << "  outer loop" << "\n";
      stlout << "    vertex " << p1[0]*m << " " << p1[1]*m << " " << p1[2]*m << "\n";
      stlout << "    vertex " << p2[0]*m << " " << p2[1]*m << " " << p2[2]*m << "\n";
      stlout << "    vertex " << p3[0]*m << " " << p3[1]*m << " " << p3[2]*m << "\n";
      stlout << "  endloop" << "\n";
      stlout << "endfacet" << "\n";
    }
    stlout << "endsolid" << endl;
  }

  void VoxelSTLGeometry :: WriteMeshsizeFile(shared_ptr<VoxelSTLGeometry> otherGeo, const string & name)
  {
    std::remove((name + ".msz").c_str());
    auto output = make_shared<spdlog::logger>("meshsize_output_" + name,
                                              make_shared<spdlog::sinks::simple_file_sink_mt>(name + ".msz"));
    output->set_level(spdlog::level::info);
    output->set_pattern("%v");
    output->info(std::to_string(otherGeo->GetVertices().Size()) + "\n");
#pragma omp parallel for
    for (size_t i = 0; i < otherGeo->GetVertices().Size(); i++)
      {
        Vertex* closest = nullptr;
        Vertex* other = &*(otherGeo->GetVertices()[i]);
        double dist = FindClosest(other, &closest);
        log->trace("Vertex: (" +
                   std::to_string(other->xy[0]*m) + "," +
                   std::to_string(other->xy[1]*m) + "," +
                   std::to_string(other->xy[2]*m) + "," +
                   ")\n" +
                   "Dist to closest: " + std::to_string(dist*m) + "\n " +
                   "Closest vertex: (" +
                   std::to_string(closest->xy[0]*m) + "," +
                   std::to_string(closest->xy[1]*m) + "," +
                   std::to_string(closest->xy[2]*m) + "," +
                   ")\n");
        output->info(std::to_string(other->xy[0]*m) + " " +
                     std::to_string(other->xy[1]*m) + " " +
                     std::to_string(other->xy[2]*m) + " " +
                     std::to_string(dist*m) + "\n");
      }
    output->info("0\n");
    output->flush();
  }

  double VoxelSTLGeometry :: FindClosest(Vertex* vertex, Vertex** closest)
  {
    double mindist = std::numeric_limits<double>::max();
    for (auto i : Range(vertices.Size()))
      {
        Vertex* v = &*vertices[i];
        Vec<3> diff = v->xy - vertex->xy;
        double len = sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]);
        if (len<mindist)
          {
            mindist = len;
            *closest = v;
          }
      }
    return mindist;
  }

  void VoxelSTLGeometry :: GenerateTrianglesAndVertices()
  {
    log->debug("Create triangles and vertices with " + std::to_string(num_threads)  + " threads...");
    Array<unique_ptr<Triangle>> triangles_thread[num_threads];
    Array<unique_ptr<Vertex>> vertices_thread[num_threads];
    Array<Vertex*> openVertices[num_threads];
    Array<Vertex*> openVerticesNextThread[num_threads-1];


    // build triangles up to 3 voxels near thread interface
#pragma omp parallel for
    for (int thread_num = 0; thread_num < num_threads; thread_num++)
      {
        for (int ix=(data->Getnx()-1)*(thread_num)/(num_threads);ix<(data->Getnx()-1)*(thread_num+1)/(num_threads)-3; ix++)
          {
            for (int iy=0; iy<data->Getny()-1; iy++){
              for (int iz=0; iz<data->Getnz()-1; iz++){
                GenerateTVCube(true, ix,iy,iz,openVertices[thread_num],triangles_thread[thread_num], vertices_thread[thread_num]);
              }
            }
          }
      }

    for (auto thread_num : Range(num_threads))
      log->debug("First openVertices for thread " + std::to_string(thread_num) + ": " + std::to_string(openVertices[thread_num].Size()));
    for (int thread_num = 1; thread_num < num_threads; thread_num++)
      for (auto i : Range(openVertices[thread_num].Size()))
        openVertices[thread_num-1].Append(openVertices[thread_num][i]);

    for (auto thread_num : Range(num_threads))
      log->debug("OpenVertices for thread " + std::to_string(thread_num) + ", after appending open vertices of next thread: " + std::to_string(openVertices[thread_num].Size()));
    // build triangles on thread interface
#pragma omp parallel for
    for (int thread_num = 0; thread_num < num_threads; thread_num++)
      for (int ix = (data->Getnx()-1)*(thread_num+1)/(num_threads)-3; ix < (data->Getnx()-1)*(thread_num+1)/(num_threads); ix++)
        for(auto iy : Range(data->Getny()-1))
          for(auto iz : Range(data->Getnz()-1))
            GenerateTVCube(false,ix,iy,iz,openVertices[thread_num],triangles_thread[thread_num],vertices_thread[thread_num]);

    for (int thread_num = 0; thread_num < num_threads; thread_num++)
      {
#pragma omp parallel sections
        {
#pragma omp section
          for (auto i : Range(triangles_thread[thread_num].Size()))
            triangles.Append(std::move(triangles_thread[thread_num][i]));
#pragma omp section
          for (auto i : Range(vertices_thread[thread_num].Size()))
            vertices.Append(std::move(vertices_thread[thread_num][i]));
        }
      }
  }

  void VoxelSTLGeometry :: PartitionVertices()
  {
    log->info("Start partitioning of vertices...");
    vertex_clustering.SetSize(0);
    Array<Vertex*> openVerticesThisLevel(vertices.Size()); // just reserve the size
    Array<Vertex*> openVerticesNextLevel(vertices.Size());
    openVerticesThisLevel.SetSize(0);
    for (size_t i = 0; i < vertices.Size(); i++)
      openVerticesNextLevel[i] = &* vertices[i];
    int level = 0;
    while(openVerticesNextLevel.Size())
      {
        for(auto v : openVerticesNextLevel)
          openVerticesThisLevel.Append(v);
        openVerticesNextLevel.SetSize(0);
        vertex_clustering.Append(make_unique<Array<Vertex*>>(openVerticesThisLevel.Size()));
        vertex_clustering[level]->SetSize(0);
        while(openVerticesThisLevel.Size())
          {
            Vertex* vert = openVerticesThisLevel[0];
            vertex_clustering[level]->Append(vert);
            openVerticesThisLevel.DeleteElement(0);
            for (auto trig : vert->neighbours)
              {
                for (auto v : { trig->v1, trig->v2, trig->v3})
                  {
                    if (openVerticesThisLevel.Contains(v))
                      {
                        openVerticesThisLevel.DeleteElement(openVerticesThisLevel.Pos(v));
                        openVerticesNextLevel.Append(v);
                      }
                    // for (auto outer_trig : v->neighbours)
                    //   {
                    //     for (auto outer_v : { outer_trig->OtherVertices(v,1),
                    //           outer_trig->OtherVertices(v,2)})
                    //       {
                    //         if (openVerticesThisLevel.Contains(outer_v))
                    //           {
                    //             openVerticesThisLevel.DeleteElement(openVerticesThisLevel.Pos(outer_v));
                    //             openVerticesNextLevel.Append(outer_v);
                    //           }
                    //       }
                    //   }
                  }
              }
          }
        level++;
      }
    log->debug("Using " + std::to_string(vertex_clustering.Size()) + " clusters for the vertices");
    log->debug("In total there are " + std::to_string(vertices.Size()) + " vertices");
    log->debug(" and " + std::to_string(triangles.Size()) + " triangles.");
    for (auto i : Range(vertex_clustering.Size()))
      log->debug("Cluster " + std::to_string(i+1) + " has " + std::to_string(vertex_clustering[i]->Size()) + " vertices.");
    log->flush();
  }

  void VoxelSTLGeometry :: SubdivisionTriangles()
  {
    double smaller_than = 0.8;
    //first step: determine triangles to be divided and their neighbours
    int sum = 0;
    for (int i = 0; i < triangles.Size(); i++){
      bool divide = false;
      //if cos(angle) between normals < 0.8 divide true
      if ( InnerProduct(triangles[i]->n, triangles[i]->NeighbourTriangle(triangles[i]->v1)->n)<smaller_than) divide = true;
      if ( InnerProduct(triangles[i]->n, triangles[i]->NeighbourTriangle(triangles[i]->v2)->n)<smaller_than) divide = true;
      if ( InnerProduct(triangles[i]->n, triangles[i]->NeighbourTriangle(triangles[i]->v3)->n)<smaller_than) divide = true;
      if(divide){
        sum+= triangles[i]->SetSubdivisionParameter();
      }
    }
    /*
     * //bad hack solution, but otherwise there could be some errors
     * //in the normals so just do it:
     *for (int i = 0; i<triangles.Size(); i++){
     *        if (triangles[i]->subdivisionParameter > 0){
     *                triangles[i]->v1->SetRatio(0.5);
     *                triangles[i]->v2->SetRatio(0.5);
     *                triangles[i]->v3->SetRatio(0.5);
     *        }
     *}
     */
    //second step: create vertices and make new triangles
    Array<Vertex*> openVertices;
    Array<unique_ptr<Vertex>> newVertices;
    int trianglesOriginalSize = triangles.Size();
    for (int i = 0; i < trianglesOriginalSize; i++){

      if (triangles[i]->subdivisionParameter == 2){
        // store v1, v2, v3 in array for better coding
        Array<Vertex*> v(3);
        v[0] = triangles[i]->v1;
        v[1] = triangles[i]->v2;
        v[2] = triangles[i]->v3;
        Array<Vertex*> vertex(3);
        for (int j = 0; j<3; j++) vertex[j] = NULL;
        triangles[i]->MakeSubdivisionVertex(openVertices,vertex,newVertices);
        triangles.Append(make_unique<Triangle>(v[0], vertex[2], vertex[1], true));
        auto tri1 = &*triangles[triangles.Size()-1];
        triangles.Append(make_unique<Triangle>(v[1], vertex[0], vertex[2], true));
        auto tri2 = &*triangles[triangles.Size()-1];
        triangles.Append(make_unique<Triangle>(v[2], vertex[1], vertex[0], true));
        auto tri3 = &*triangles[triangles.Size()-1];
        triangles.Append(make_unique<Triangle>(vertex[0], vertex[1], vertex[2], true));
        auto tri4 = &*triangles[triangles.Size()-1];

        vertex[0]->addNeighbour(tri2);
        vertex[0]->addNeighbour(tri3);
        vertex[0]->addNeighbour(tri4);
        vertex[1]->addNeighbour(tri1);
        vertex[1]->addNeighbour(tri3);
        vertex[1]->addNeighbour(tri4);
        vertex[2]->addNeighbour(tri1);
        vertex[2]->addNeighbour(tri2);
        vertex[2]->addNeighbour(tri4);
        v[0]->DeleteNeighbour(&*triangles[i]);
        v[0]->addNeighbour(tri1);
        v[1]->DeleteNeighbour(&*triangles[i]);
        v[1]->addNeighbour(tri2);
        v[2]->DeleteNeighbour(&*triangles[i]);
        v[2]->addNeighbour(tri3);
        Triangle* n1;
        for (int j = 0; j<3; j++){
          n1 = triangles[i]->NeighbourTriangle(v[j]);
          if(n1!=NULL && n1->subdivisionParameter == 1){
            //find vertex which is not on
            //triangles[i]:
            Vertex* visavisVertex;
            if(!triangles[i]->doIhave(n1->v1)) {visavisVertex = n1->v1;}
            if(!triangles[i]->doIhave(n1->v2)) {visavisVertex = n1->v2;}
            if(!triangles[i]->doIhave(n1->v3)) {visavisVertex = n1->v3;}
            triangles.Append(make_unique<Triangle>(visavisVertex, n1->OtherVertices(visavisVertex,1), vertex[j],true));
            auto tri11 = &* triangles[triangles.Size()-1];
            triangles.Append(make_unique<Triangle>(visavisVertex, vertex[j], n1->OtherVertices(visavisVertex,2),true));
            auto tri12 = &* triangles[triangles.Size()-1];
            vertex[j]->addNeighbour(tri11);
            vertex[j]->addNeighbour(tri12);
            visavisVertex->DeleteNeighbour(n1);
            visavisVertex->addNeighbour(tri11);
            visavisVertex->addNeighbour(tri12);
            n1->OtherVertices(visavisVertex,1)->DeleteNeighbour(n1);
            n1->OtherVertices(visavisVertex,1)->addNeighbour(tri11);
            n1->OtherVertices(visavisVertex,2)->DeleteNeighbour(n1);
            n1->OtherVertices(visavisVertex,2)->addNeighbour(tri12);
          }
        }

      }
    }
    for (int i=trianglesOriginalSize-1; i>=0; i--){
      if (triangles[i]->subdivisionParameter > 0) {
        triangles.DeleteElement(i);
      }
    }
    for (int i = 0; i<newVertices.Size(); i++){
      vertices.Append(move(newVertices[i]));
    }

    //    QuickSortVertices(0, vertices.Size());
    //return sum;

  }

  void VoxelSTLGeometry :: MinimizeEnergy(double weight_area, double weight_minimum)
  {
    double e = 0; double e_diff = 0;
    log->info("Minimize energy with " + std::to_string(num_threads) + " threads");
    for (int cluster = 0; cluster<vertex_clustering.Size(); cluster++)
      {
#pragma omp parallel for
        for (size_t i = 0; i < vertex_clustering[cluster]->Size();i++)
          {
            auto vertex = (*vertex_clustering[cluster])[i];
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
            std::set<Vertex*> dofs;
            std::set<Vertex*>::iterator dof_it;
            std::set<Triangle*> tris;
            std::set<Triangle*>::iterator tri_it;
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
                Triangle* me;
                Triangle* he;
                Vertex* me_v1;
                Vertex* me_v2;
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

  void VoxelSTLGeometry :: GenerateTVCube(bool deletevertices,int x, int y, int z, Array<Vertex*>& openVertices, Array<unique_ptr<Triangle>>& triangles, Array<unique_ptr<Vertex>>& vertices)
  {
    int one = 1;
    //flip cube diagonal every second step to match faces of tetraeders inside
    if((x+y+z)%2 == 0)
      {
        x++; y++; z++; one=-1;
      }
    bool p000 = isUsedVoxel(x,y,z);
    bool p100 = isUsedVoxel(x+one,y,z);
    bool p010 = isUsedVoxel(x,y+one,z);
    bool p001 = isUsedVoxel(x,y,z+one);
    bool p110 = isUsedVoxel(x+one,y+one,z);
    bool p101 = isUsedVoxel(x+one,y,z+one);
    bool p011 = isUsedVoxel(x,y+one,z+one);
    bool p111 = isUsedVoxel(x+one,y+one,z+one);
    if (!p000 && !p100 && !p010 && !p001 && !p110 && !p101 && !p011 && !p111)
      return;

    int px,py,pz;
    int count = 0;
    // first tetraeder
    Array<Vertex*> tet1;
    if(p000-p100) {
      tet1 += MakeVertex(deletevertices,vertices,openVertices,x,y,z,x+one,y,z);
      count++;
      if(!p000){px=x;py=y;pz=z;}
      else{px=x+one;py=y;pz=z;}
    }
    if(p000-p010) {
      tet1 += MakeVertex(deletevertices,vertices,openVertices,x,y,z,x,y+one,z);
      count++;
      if(!p000){px=x;py=y;pz=z;}
      else{px=x;py=y+one;pz=z;}
    }
    if(p000-p001) {
      tet1 += MakeVertex(deletevertices,vertices,openVertices,x,y,z,x,y,z+one);
      count++;
      if(!p000){px=x;py=y;pz=z;}
      else{px=x;py=y;pz=z+one;}
    }
    if(p100-p010) {
      tet1 += MakeVertex(deletevertices,vertices,openVertices,x+one,y,z,x,y+one,z);
      count++;
      if(!p100){px=x+one;py=y;pz=z;}
      else{px=x;py=y+one;pz=z;}
    }
    if(p100-p001) {
      tet1 += MakeVertex(deletevertices,vertices,openVertices,x+one,y,z,x,y,z+one);
      count++;
      if(!p100){px=x+one;py=y;pz=z;}
      else{px=x;py=y;pz=z+one;}
    }
    if(p010-p001) {
      tet1 += MakeVertex(deletevertices,vertices,openVertices,x,y+one,z,x,y,z+one);
      count++;
      if(!p010){px=x;py=y+one;pz=z;}
      else{px=x;py=y;pz=z+one;}
    }
    if(count >= 3) {
      auto triangle11 = make_unique<Triangle>(tet1[0],tet1[1],tet1[2]);
      for (int i=0; i<3; i++){
        tet1[i]->addNeighbour(&*triangle11);
      }
      triangles.Append(std::move(triangle11));
    }
    if(count==4){
      auto triangle12 = make_unique<Triangle>(tet1[1],tet1[2],tet1[3]);
      for (int i=1; i<4; i++){
        tet1[i]->addNeighbour(&*triangle12);
      }
      triangles.Append(std::move(triangle12));
    }

    // second tetraeder
    Array<Vertex*> tet2;
    count = 0;
    if(p100-p110) {
      tet2 += MakeVertex(deletevertices,vertices,openVertices,x+one,y,z,x+one,y+one,z);
      count++;
      if(!p100){px=x+one;py=y;pz=z;}
      else{px=x+one;py=y+one;pz=z;}
    }
    if(p110-p111) {
      tet2 += MakeVertex(deletevertices,vertices,openVertices,x+one,y+one,z,x+one,y+one,z+one);
      count++;
      if(!p110){px=x+one;py=y+one;pz=z;}
      else{px=x+one;py=y+one;pz=z+one;}
    }
    if(p100-p111) {
      tet2 += MakeVertex(deletevertices,vertices,openVertices,x+one,y,z,x+one,y+one,z+one);
      count++;
      if(!p100){px=x+one;py=y;pz=z;}
      else{px=x+one;py=y+one;pz=z+one;}
    }
    if(p010-p110) {
      tet2 += MakeVertex(deletevertices,vertices,openVertices,x,y+one,z,x+one,y+one,z);
      count++;
      if(!p010){px=x;py=y+one;pz=z;}
      else{px=x+one;py=y+one;pz=z;}
    }
    if(p100-p010) {
      tet2 += MakeVertex(deletevertices,vertices,openVertices,x+one,y,z,x,y+one,z);
      count++;
      if(!p100){px=x+one;py=y;pz=z;}
      else{px=x;py=y+one;pz=z;}
    }
    if(p010-p111) {
      tet2 += MakeVertex(deletevertices,vertices,openVertices,x,y+one,z,x+one,y+one,z+one);
      count++;
      if(!p010){px=x;py=y+one;pz=z;}
      else{px=x+one;py=y+one;pz=z+one;}
    }
    if(count >= 3) {
      auto triangle21 = make_unique<Triangle>(tet2[0],tet2[1],tet2[2]);
      for (int i=0; i<3; i++){
        tet2[i]->addNeighbour(&*triangle21);
      }
      triangles.Append(move(triangle21));
    }
    if(count==4){
      auto triangle22 = make_unique<Triangle>(tet2[1],tet2[2],tet2[3]);
      for (int i=1; i<4; i++){
        tet2[i]->addNeighbour(&*triangle22);
      }
      triangles.Append(move(triangle22));
    }
    // third tetraeder
    Array<Vertex*> tet3;
    count = 0;
    if(p100-p001) {
      tet3 += MakeVertex(deletevertices,vertices,openVertices,x+one,y,z,x,y,z+one);
      count++;
      if(!p100){px=x+one;py=y;pz=z;}
      else{px=x;py=y;pz=z+one;}
    }
    if(p100-p101) {
      tet3 += MakeVertex(deletevertices,vertices,openVertices,x+one,y,z,x+one,y,z+one);
      count++;
      if(!p100){px=x+one;py=y;pz=z;}
      else{px=x+one;py=y;pz=z+one;}
    }
    if(p001-p101) {
      tet3 += MakeVertex(deletevertices,vertices,openVertices,x,y,z+one,x+one,y,z+one);
      count++;
      if(!p001){px=x;py=y;pz=z+one;}
      else{px=x+one;py=y;pz=z+one;}
    }
    if(p100-p111) {
      tet3 += MakeVertex(deletevertices,vertices,openVertices,x+one,y,z,x+one,y+one,z+one);
      count++;
      if(!p100){px=x+one;py=y;pz=z;}
      else{px=x+one;py=y+one;pz=z+one;}
    }
    if(p001-p111) {
      tet3 += MakeVertex(deletevertices,vertices,openVertices,x,y,z+one,x+one,y+one,z+one);
      count++;
      if(!p001){px=x;py=y;pz=z+one;}
      else{px=x+one;py=y+one;pz=z+one;}
    }
    if(p101-p111) {
      tet3 += MakeVertex(deletevertices,vertices,openVertices,x+one,y,z+one,x+one,y+one,z+one);
      count++;
      if(!p101){px=x+one;py=y;pz=z+one;}
      else{px=x+one;py=y+one;pz=z+one;}
    }
    if(count >= 3) {
      auto triangle31 = make_unique<Triangle>(tet3[0],tet3[1],tet3[2]);
      for (int i=0; i<3; i++){
        tet3[i]->addNeighbour(&*triangle31);
      }
      triangles.Append(move(triangle31));
    }
    if(count==4){
      auto triangle32 = make_unique<Triangle>(tet3[1],tet3[2],tet3[3]);
      for (int i=1; i<4; i++){
        tet3[i]->addNeighbour(&*triangle32);
      }
      triangles.Append(move(triangle32));
    }

    // fourth tetraeder
    Array<Vertex*> tet4;
    count = 0;
    if(p001-p010) {
      tet4 += MakeVertex(deletevertices,vertices,openVertices,x,y,z+one,x,y+one,z);
      count++;
      if(!p001){px=x;py=y;pz=z+one;}
      else{px=x;py=y+one;pz=z;}
    }
    if(p001-p011) {
      tet4 += MakeVertex(deletevertices,vertices,openVertices,x,y,z+one,x,y+one,z+one);
      count++;
      if(!p001){px=x;py=y;pz=z+one;}
      else{px=x;py=y+one;pz=z+one;}
    }
    if(p010-p011) {
      tet4 += MakeVertex(deletevertices,vertices,openVertices,x,y+one,z,x,y+one,z+one);
      count++;
      if(!p010){px=x;py=y+one;pz=z;}
      else{px=x;py=y+one;pz=z+one;}
    }
    if(p001-p111) {
      tet4 += MakeVertex(deletevertices,vertices,openVertices,x,y,z+one,x+one,y+one,z+one);
      count++;
      if(!p001){px=x;py=y;pz=z+one;}
      else{px=x+one;py=y+one;pz=z+one;}
    }
    if(p010-p111) {
      tet4 += MakeVertex(deletevertices,vertices,openVertices,x,y+one,z,x+one,y+one,z+one);
      count++;
      if(!p010){px=x;py=y+one;pz=z;}
      else{px=x+one;py=y+one;pz=z+one;}
    }
    if(p011-p111) {
      tet4 += MakeVertex(deletevertices,vertices,openVertices,x,y+one,z+one,x+one,y+one,z+one);
      count++;
      if(!p011){px=x;py=y+one;pz=z+one;}
      else{px=x+one;py=y+one;pz=z+one;}
    }
    if(count >= 3) {
      auto triangle41 = make_unique<Triangle>(tet4[0],tet4[1],tet4[2]);
      for (int i=0; i<3; i++){
        tet4[i]->addNeighbour(&*triangle41);
      }
      triangles.Append(move(triangle41));
    }
    if(count==4){
      auto triangle42 = make_unique<Triangle>(tet4[1],tet4[2],tet4[3]);
      for (int i=1; i<4; i++){
        tet4[i]->addNeighbour(&*triangle42);
      }
      triangles.Append(move(triangle42));
    }
    // fifth tetraeder
    Array<Vertex*> tet5;
    count = 0;
    if(p100-p001) {
      tet5 += MakeVertex(deletevertices,vertices,openVertices,x+one,y,z,x,y,z+one);
      count++;
      if(!p100){px=x+one;py=y;pz=z;}
      else{px=x;py=y;pz=z+one;}
    }
    if(p100-p010) {
      tet5 += MakeVertex(deletevertices,vertices,openVertices,x+one,y,z,x,y+one,z);
      count++;
      if(!p100){px=x+one;py=y;pz=z;}
      else{px=x;py=y+one;pz=z;}
    }
    if(p010-p001) {
      tet5 += MakeVertex(deletevertices,vertices,openVertices,x,y+one,z,x,y,z+one);
      count++;
      if(!p010){px=x;py=y+one;pz=z;}
      else{px=x;py=y;pz=z+one;}
    }
    if(p100-p111) {
      tet5 += MakeVertex(deletevertices,vertices,openVertices,x+one,y,z,x+one,y+one,z+one);
      count++;
      if(!p100){px=x+one;py=y;pz=z;}
      else{px=x+one;py=y+one;pz=z+one;}
    }
    if(p001-p111) {
      tet5 += MakeVertex(deletevertices,vertices,openVertices,x,y,z+one,x+one,y+one,z+one);
      count++;
      if(!p001){px=x;py=y;pz=z+one;}
      else{px=x+one;py=y+one;pz=z+one;}
    }
    if(p010-p111) {
      tet5 += MakeVertex(deletevertices,vertices,openVertices,x,y+one,z,x+one,y+one,z+one);
      count++;
      if(!p010){px=x;py=y+one;pz=z;}
      else{px=x+one;py=y+one;pz=z+one;}
    }
    if(count >= 3) {
      auto triangle51 = make_unique<Triangle>(tet5[0],tet5[1],tet5[2]);
      for (int i=0; i<3; i++){
        tet5[i]->addNeighbour(&*triangle51);
      }
      triangles.Append(move(triangle51));
    }
    if(count==4){
      auto triangle52 = make_unique<Triangle>(tet5[1],tet5[2],tet5[3]);
      for (int i=1; i<4; i++){
        tet5[i]->addNeighbour(&*triangle52);
      }
      triangles.Append(move(triangle52));
    }
  }

  // create new vertex and store unique_ptr to it in vertices, put pointer in openVertices and return
  // pointer to that new vertex
  VoxelSTLGeometry::Vertex* VoxelSTLGeometry :: MakeVertex(bool deletevertices, Array<unique_ptr<Vertex>> & vertices, Array<Vertex*> & openVertices, int x1, int y1, int z1, int x2, int y2, int z2)
  {
    if (isUsedVoxel(x1,y1,z1)) {
      //nothing to be done, x is inner point
    }
    else{
      int tmp = x1; x1 = x2; x2 = tmp;
      tmp = y1; y1 = y2; y2 = tmp;
      tmp = z1; z1 = z2; z2 = tmp;
    }

    Vec<3,double> x(x1,y1,z1);
    Vec<3,double> y(x2,y2,z2);
    for (int i = openVertices.Size()-1; i>=0; i--){
      if (openVertices[i]->doIhave(x,y)) {
        return (openVertices[i]);
      }
      double open_x = openVertices[i]->x[0];
      if (deletevertices && open_x<x1-3 && open_x<x2-3){
        openVertices.DeleteElement(i);
      }
    }
    auto newVertex = make_unique<Vertex>(x,y);
    auto pnewV = &*newVertex;
    openVertices.Append(pnewV);
    vertices.Append(move(newVertex));
    return pnewV;
  }

  void VoxelSTLGeometry::Triangle::MakeSubdivisionVertex(Array<Vertex*>& openVertices, Array<Vertex*>& vertex, Array<unique_ptr<Vertex>>& newVertices){
    bool deleteVertex = false;
    for(int i = openVertices.Size()-1; i>=0; i--){
      deleteVertex = false;
      bool go = true;
      for (int j= 0; j<3 && go; j++){
        if (abs(openVertices[i]->x[j] - 0.5*(v1->x[j] + v2->x[j]))>1e-7 || abs(openVertices[i]->y[j] - 0.5*(v1->y[j] + v2->y[j]))>1e-7) {
          go = false;
        }
      }
      if(go){
        vertex[2]=openVertices[i];
        deleteVertex == true;
      }
      go = !deleteVertex ;
      for (int j= 0; j<3 && go; j++){
        if (abs(openVertices[i]->x[j] - 0.5*(v1->x[j] + v3->x[j]))>1e-7 || abs(openVertices[i]->y[j] - 0.5*(v1->y[j] + v3->y[j]))>1e-7) {
          go = false;
        }
      }
      if(go){
        vertex[1]=openVertices[i];
        deleteVertex == true;
      }
      go = !deleteVertex;
      for (int j= 0; j<3 && go; j++){
        if (abs(openVertices[i]->x[j] - 0.5*(v2->x[j] + v3->x[j]))>1e-7 || abs(openVertices[i]->y[j] - 0.5*(v2->y[j] + v3->y[j]))>1e-7) {
          go = false;
        }
      }
      if(go){
        vertex[0]=openVertices[i];
        deleteVertex == true;
      }
      if(deleteVertex){
        openVertices.DeleteElement(i);
      }
    }
    if(vertex[2] == NULL){
      auto newv = make_unique<Vertex>(v1,v2,n);
      vertex[2] = &*newv;
      openVertices.Append(&*newv);
      newVertices.Append(move(newv));
    }
    if(vertex[1] == NULL){
      auto newv = make_unique<Vertex>(v1,v3,n);
      vertex[1] = &* newv;
      openVertices.Append(&*newv);
      newVertices.Append(move(newv));
    }
    if(vertex[0] == NULL){
      auto newv = make_unique<Vertex>(v2,v3,n);
      vertex[0] = &*newv;
      openVertices.Append(&*newv);
      newVertices.Append(move(newv));
    }
    for (int i = 0; i<3; i++){
      for(int j = 0; j<3; j++){
        if (i!=j && vertex[i] == vertex[j]) cout << "error---------------------------------------------" << endl;
      }
    }
  }

  void VoxelSTLGeometry :: QuickSortVertices(size_t low, size_t up)
  {
    if (up-low < 2) return;
    auto i = low;
    auto j = up-1;
    auto & midval = *vertices[(i+j)/2];
    do
      {
        while (vertices[i]->x[0] < midval.x[0]) i++;
        while (midval.x[0] < vertices[j]->x[0]) j--;
        auto tmp = move(vertices[i]);
        vertices[i] = move(vertices[j]);
        vertices[j] = move(tmp);
        i++; j--;
      } while(i <= j);
    QuickSortVertices(low,j+1);
    QuickSortVertices(i,up);
  }

  double Newton(std::function<double (double)>  func, double r0, double &energydifference, double & energy){
	double r=r0;
	energy = func(r0);
	//newton method
	int count = 0;
	double alpha = 1.0;
	int countalpha = 1;
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

}
