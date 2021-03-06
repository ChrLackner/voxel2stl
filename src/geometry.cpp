
#include "voxel2stl.hpp"
#include <omp.h>
#include <set>
#include <assert.h>

namespace std
{
  template<int N>
  string to_string(ngbla::Vec<N> vec)
  {
    string str = "(" + to_string(vec[0]);
    for(auto i : ngcore::Range(1,N))
      str += ", " + to_string(vec[i]);
    str += ")";
    return str;
  }
}

namespace voxel2stl
{
  VoxelSTLGeometry::VoxelSTLGeometry(shared_ptr<VoxelData> adata,
                                     const Array<size_t>& amaterials,
                                     const Array<size_t>& aboundaries)
    : data(adata), materials(amaterials), boundaries(aboundaries)
  {
    if(!adata)
      return;
    m = data->Getm();
#pragma omp parallel
    {
      num_threads = omp_get_num_threads();
    }
    GenerateTrianglesAndVertices();
    PartitionVertices();
  }

  void VoxelSTLGeometry :: WriteSTL(string filename)
  {
    // log->info("Start writing to stl file");
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
          // log->error("Error in triangle!");
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
    // log->info("Done");
  }

  void VoxelSTLGeometry :: CutOffAt(int component, bool positive, double val)
  {
    double eps = 1e-7;
    auto isOutside = [component, positive, val, eps] (Vertex* v)
                     { return positive ? v->xy[component] > val-eps : v->xy[component] < val+eps; };
    std::set<size_t> to_remove_trig;
    std::set<Vertex*> to_remove_vert;
    std::set<Vertex*> moved_verts;
    Array<std::pair<Vertex*,Vertex*>> verts_for_new_trigs;
    for(auto i : Range(triangles.Size()))
      {
        auto& trig = triangles[i];
        bool out[3] = { isOutside(trig->v1), isOutside(trig->v2), isOutside(trig->v3) };
        int nout = std::count(std::begin(out), std::end(out), true);
        if(nout == 3)
          {
            to_remove_trig.insert(i);
            for(auto v : {trig->v1, trig->v2, trig->v3})
              to_remove_vert.insert(v);
          }
        if(nout == 2 || nout == 1)
          {
            if(out[0])
              {
                trig->v1->xy[component] = positive ? val + eps : val - eps;
                moved_verts.insert(trig->v1);
              }
            if(out[1])
              {
                trig->v2->xy[component] = positive ? val + eps : val - eps;
                moved_verts.insert(trig->v2);
              }
            if(out[2])
              {
                trig->v3->xy[component] = positive ? val + eps : val - eps;
                moved_verts.insert(trig->v3);
              }
          }
        if (nout == 2)
          {
            if(!out[0])
              verts_for_new_trigs.Append(std::make_pair(trig->v3,trig->v2));
            if(!out[1])
              verts_for_new_trigs.Append(std::make_pair(trig->v1,trig->v3));
            if(!out[2])
              verts_for_new_trigs.Append(std::make_pair(trig->v2,trig->v1));
          }
      }
    for(auto v : moved_verts)
        v->xy[component] -= positive ? eps : -eps;
    for(auto it = to_remove_trig.rbegin(); it != to_remove_trig.rend(); ++it)
        triangles.DeleteElement(*it);
    while(verts_for_new_trigs.Size())
      {
        auto mid_vertex = make_unique<Vertex>();
        mid_vertex->xy = 0.;
        // collect vertices belonging to one cut face
        Array<std::pair<Vertex*,Vertex*>> pairs;
        pairs.Append(std::move(verts_for_new_trigs.Last()));
        verts_for_new_trigs.DeleteLast();
        while(true)
          {
            size_t index; bool found = false;
            for(size_t i = 0; i < verts_for_new_trigs.Size(); i++)
              if(verts_for_new_trigs[i].first == pairs.Last().second)
                {
                  found = true;
                  index = i;
                }
            if(!found)
              break;
            pairs.Append(std::move(verts_for_new_trigs[index]));
            verts_for_new_trigs.DeleteElement(index);
          }
        if(pairs[0].first != pairs.Last().second)
          throw Exception("Cutting plane not closed!");
        for(auto& pair : pairs)
            mid_vertex->xy += pair.first->xy;
        mid_vertex->xy *= 1./pairs.Size();
        auto mid_vertex_ptr = &* mid_vertex;
        vertices.Append(std::move(mid_vertex));
        for(auto pair : pairs)
          triangles.Append(make_unique<Triangle>(pair.first, pair.second, mid_vertex_ptr));
      }
  }

  void VoxelSTLGeometry :: GenerateTrianglesAndVertices()
  {
    // log->info("Create triangles and vertices with " + std::to_string(num_threads)  + " threads...");
    Array<unique_ptr<Vertex>> thread_vertices[num_threads];
   // log->debug("Data size is " + to_string(data->Getnx()) + " x " + to_string(data->Getny()) + " x " +
   //             to_string(data->Getnz()));
   // log->debug("Boundaries are:\nx: " + to_string(boundaries[0]) + " to " + to_string(boundaries[1]) + "\ny: " +
   //             to_string(boundaries[2]) + " to " + to_string(boundaries[3]) + "\nz: " + to_string(boundaries[4]) +
   //             " to " + to_string(boundaries[5]));
   //  log->debug("Materials are:");
    // for(auto val : materials)
    //   log->debug(to_string(val));

    size_t sum_size = 0;
    // int count = 0;

    for(auto ixiy : {true,false})
      {
        for(auto ixiz : {true,false})
          {
#pragma omp parallel for
            for (size_t ix=max2(0,int(boundaries[0])-2); ix<min2(data->Getnx()-1, boundaries[1]+2); ix++)
              for (size_t iy=max2(0,int(boundaries[2])-2); iy<min2(data->Getny()-1, boundaries[3]+2);iy++)
                for (size_t iz = max2(0,int(boundaries[4])-2); iz < min2(data->Getnz()-1,boundaries[5]+2); iz++)
                  if((ix+iy)%2==ixiy)
                    if((ix+iz)%2==ixiz)
                      GenerateTVCube(ix, iy, iz, thread_vertices[omp_get_thread_num()]);

            for(auto i : Range(num_threads))
              sum_size += thread_vertices[i].Size();
            vertices.SetAllocSize(sum_size);
            for(auto i : Range(num_threads))
              {
                for(auto j : Range(thread_vertices[i].Size()))
                  vertices.Append(move(thread_vertices[i][j]));
                thread_vertices[i].SetSize(0);
              }
            // log->debug("Num vertices created after " + std::to_string(++count) +
            //            " round = " + std::to_string(sum_size));
          }
      }
    // for(auto i : Range(vertices.Size()))
    //   if(vertices[i]->nn <3)
    //     log->warn("Vertex has less than 3 triangles: " + to_string(vertices[i]->xy));
    // log->info("Triangles generated.");
  }

  void VoxelSTLGeometry :: PartitionVertices()
  {
    // log->info("Start partitioning of vertices...");
    Array<size_t> cluster_count;
    for (auto& vert : vertices)
      {
        while(cluster_count.Size()<vert->cluster+1)
          cluster_count.Append(0);
        cluster_count[vert->cluster]++;
      }

    for(auto i : Range(cluster_count.Size()))
      vertex_clustering.Append(make_unique<Array<Vertex*>>());

    for(auto i : Range(cluster_count.Size()))
      vertex_clustering[i]->SetAllocSize(cluster_count[i]);

    for (auto& vert : vertices)
      vertex_clustering[vert->cluster]->Append(&*vert);

    // log->debug("Using " + std::to_string(vertex_clustering.Size()) + " clusters for the vertices");
    // log->debug("In total there are " + std::to_string(vertices.Size()) + " vertices");
    // log->debug(" and " + std::to_string(triangles.Size()) + " triangles.");
    // for (auto i : Range(vertex_clustering.Size()))
    //   log->debug("Cluster " + std::to_string(i+1) + " has " + std::to_string(vertex_clustering[i]->Size()) + " vertices.");
    // log->flush();
    // log->info("Partitioning done.");
  }

  void VoxelSTLGeometry :: SubdivideTriangles()
  {
    // log->info("Start subdividing triangles");
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
    size_t trianglesOriginalSize = triangles.Size();
    for (size_t i = 0; i < trianglesOriginalSize; i++){

      if (triangles[i]->subdivisionParameter == 2){
        // store v1, v2, v3 in array for better coding
        Array<Vertex*> v(3);
        v[0] = triangles[i]->v1;
        v[1] = triangles[i]->v2;
        v[2] = triangles[i]->v3;
        Array<Vertex*> vertex(3);
        for (int j = 0; j<3; j++) vertex[j] = NULL;
        triangles[i]->MakeSubdivisionVertex(openVertices,vertex,newVertices,voxel_to_vertex);
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
        for (auto vert : vertex)
          vert->FindFreeCluster();
      }
    }
    for (size_t j=0; j<trianglesOriginalSize; j++)
      {
        size_t i = trianglesOriginalSize-1-j;
        if (triangles[i]->subdivisionParameter > 0) {
          triangles.DeleteElement(i);
      }
    }
    Array<size_t> cluster_count;
    for (auto& vert : newVertices)
      {
        while(cluster_count.Size()<vert->cluster+1)
          cluster_count.Append(0);
        cluster_count[vert->cluster]++;
      }

    while(cluster_count.Size() > vertex_clustering.Size())
      vertex_clustering.Append(make_unique<Array<Vertex*>>());

    for(auto i : Range(cluster_count.Size()))
      vertex_clustering[i]->SetAllocSize(vertex_clustering[i]->Size() + cluster_count[i]);

    for (auto& vert : newVertices)
      {
        vertex_clustering[vert->cluster]->Append(&*vert);
        vertices.Append(move(vert));
      }
    // log->info("done.");
  }

  void VoxelSTLGeometry :: GenerateTVCube(size_t x, size_t y, size_t z,
                                          Array<unique_ptr<Vertex>>& thread_vertices)
  {
    // SPDLOG_DEBUG(log, "Start cube (" + to_string(x) + "," + to_string(y) + "," + to_string(z) + ")");
    int one = 1;
    //flip cube diagonal every second step to match faces of tetraeders inside
    size_t x_old = x; size_t y_old = y; size_t z_old = z;
    if((x+y+z)%2 == 0)
      {
        x++; y++; z++; one=-1;
      }

    bool isused[8] = {isUsedVoxel(x,y,z),isUsedVoxel(x,y,z+one),isUsedVoxel(x,y+one,z),
                      isUsedVoxel(x,y+one,z+one),isUsedVoxel(x+one,y,z),isUsedVoxel(x+one,y,z+one),
                      isUsedVoxel(x+one,y+one,z),isUsedVoxel(x+one,y+one,z+one)};
    // SPDLOG_DEBUG(log, "Is used = ");
    // for(auto i : Range(8))
    //   SPDLOG_DEBUG(log,to_string(isused[i]));
    if (std::all_of(std::begin(isused),std::end(isused),[](bool b) { return !b; }))
      return;
    // SPDLOG_DEBUG(log, "Not all are unused");
    unsigned char combinations[5][6][2] = {{{0,4},{0,2},{0,1},{4,2},{4,1},{2,1}},
                                           {{4,6},{6,7},{4,7},{2,6},{4,2},{2,7}},
                                           {{4,1},{4,5},{1,5},{4,7},{1,7},{5,7}},
                                           {{1,2},{1,3},{2,3},{1,7},{2,7},{3,7}},
                                           {{4,1},{4,2},{2,1},{4,7},{1,7},{2,7}}};
    for(auto& tets : combinations)
      {
        Array<Vertex*> tet(4);
        tet.SetSize0();
        size_t count = 0;
        for(auto i : tets)
          {
            if(isused[i[0]] - isused[i[1]])
              {
                auto nvert = MakeVertex(thread_vertices,
                                        x+one*bool(i[0]&4),y+one*bool(i[0]&2),z+one*bool(i[0]&1),
                                        x+one*bool(i[1]&4),y+one*bool(i[1]&2),z+one*bool(i[1]&1));
                assert(nvert);
                tet.Append(nvert);
                count++;
              }
          }
        // SPDLOG_DEBUG(log, "Found " + to_string(count) + " cuttings in voxel " + to_string(x) + ", " +
        //              to_string(y) + ", " + to_string(z));
        if(count >= 3) {
          auto triangle1 = make_unique<Triangle>(tet[0],tet[1],tet[2]);
          // SPDLOG_DEBUG(log,"Created new triangle1: " + triangle1->to_string());
          for (int i=0; i<3; i++){
            tet[i]->addNeighbour(&*triangle1);
          }
#pragma omp critical(triangles)
          { triangles.Append(std::move(triangle1)); }
        }
        if(count==4){
          auto triangle2 = make_unique<Triangle>(tet[1],tet[2],tet[3]);
          // SPDLOG_DEBUG(log,"Created new triangle2: " + triangle2->to_string());
          for (int i=1; i<4; i++){
            tet[i]->addNeighbour(&*triangle2);
          }
#pragma omp critical(triangles)
          { triangles.Append(std::move(triangle2)); }
        }
        for(auto i : Range(count))
          {
            auto vert = tet[i];
            vert->FindFreeCluster();
          }
      }
  }
  // create new vertex and store unique_ptr to it in vertices, put pointer in openVertices and return
  // pointer to that new vertex
  
  VoxelSTLGeometry::Vertex* VoxelSTLGeometry ::
  MakeVertex(Array<unique_ptr<Vertex>>& thread_vertices,
             size_t x1, size_t y1, size_t z1, size_t x2, size_t y2, size_t z2)
  {
    if (isUsedVoxel(x1,y1,z1)) {
      //nothing to be done, x is inner point
    }
    else{
      size_t tmp = x1; x1 = x2; x2 = tmp;
      tmp = y1; y1 = y2; y2 = tmp;
      tmp = z1; z1 = z2; z2 = tmp;
    }

    Vec<3,double> x(x1,y1,z1);
    Vec<3,double> y(x2,y2,z2);


    v2v_map::iterator itV;
    // if we don't do this critically another thread may change the map in the meantime
#pragma omp critical(voxel_to_vertex)
    itV = voxel_to_vertex.find(make_tuple(x1,y1,z1,x2,y2,z2));

    if(itV != voxel_to_vertex.end())
      {
        auto vert = itV->second;
        // SPDLOG_DEBUG(log, "Vertex " + vert->to_string() + " found in global vertices");
        return vert;
      }

    auto newVertex = make_unique<Vertex>(x,y);
    auto nvert = &*newVertex;
    nvert->cluster = 0;
#pragma omp critical(voxel_to_vertex)
    voxel_to_vertex[make_tuple(x1,y1,z1,x2,y2,z2)] = nvert;

    // SPDLOG_DEBUG(log, "Created new vertex " + nvert->to_string());
    thread_vertices.Append(move(newVertex));
    return nvert;
  }

  void VoxelSTLGeometry::Triangle::
  MakeSubdivisionVertex(Array<Vertex*>& openVertices, Array<Vertex*>& vertex,
                        Array<unique_ptr<Vertex>>& newVertices,
                        v2v_map& vox_to_vert){
    bool deleteVertex = false;
    for(size_t j = 0; j < openVertices.Size(); j++)
      {
        size_t i = openVertices.Size()-1-j;
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
      auto newv = make_unique<Vertex>(v1,v2);
      auto vert = &*newv;
      vertex[2] = vert;
      openVertices.Append(vert);
      vox_to_vert[make_tuple(newv->x[0],newv->x[1],newv->x[2],
                             newv->y[0],newv->y[1],newv->y[2])] = &*newv;
      newVertices.Append(move(newv));
    }
    if(vertex[1] == NULL){
      auto newv = make_unique<Vertex>(v1,v3);
      auto vert = &*newv;
      vertex[1] = vert;
      openVertices.Append(vert);
      vox_to_vert[make_tuple(newv->x[0],newv->x[1],newv->x[2],
                             newv->y[0],newv->y[1],newv->y[2])] = &*newv;
      newVertices.Append(move(newv));
    }
    if(vertex[0] == NULL){
      auto newv = make_unique<Vertex>(v2,v3);
      auto vert = &*newv;
      vertex[0] = vert;
      openVertices.Append(vert);
      vox_to_vert[make_tuple(newv->x[0],newv->x[1],newv->x[2],
                             newv->y[0],newv->y[1],newv->y[2])] = &*newv;
      newVertices.Append(move(newv));
    }
    for (int i = 0; i<3; i++){
      for(int j = 0; j<3; j++){
        if (i!=j && vertex[i] == vertex[j]) cout << "error---------------------------------------------" << endl;
      }
    }
  }

}
