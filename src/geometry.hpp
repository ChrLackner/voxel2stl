#ifndef __FILE_GEOMETRY_HPP
#define __FILE_GEOMETRY_HPP

namespace voxel2stl
{
  class Smoother;

  class VoxelSTLGeometry : public pyspdlog::LoggedClass
  {
  public:
    struct Vertex;
    struct Triangle;
  protected:
    using Tclustering = Array<unique_ptr<Array<Vertex*>>>;
    shared_ptr<VoxelData> data;
    Array<unique_ptr<Vertex>> vertices;
    Array<unique_ptr<Triangle>> triangles;
    double m;
    int num_threads;
    Tclustering vertex_clustering;
    Array<size_t> materials, boundaries;
    using v2v_map = std::map<std::tuple<size_t,size_t,size_t,size_t,size_t,size_t>,Vertex*>;
    v2v_map voxel_to_vertex;

  public:
    VoxelSTLGeometry(shared_ptr<VoxelData> adata, const Array<size_t>& amaterials,
                     const Array<size_t>& aboundaries, shared_ptr<spdlog::logger> log);

    void WriteSTL(string filename);
    void SubdivideTriangles();

    Vertex* GetClosestVertex(Vec<3> p)
    {
      Vertex* vertex = GetVertex(0);
      for(auto& v : vertices)
        if(L2Norm(m * v->xy - p) < L2Norm(m * vertex->xy - p))
          vertex = &*v;
      return vertex;
    }

    auto& GetVertexClustering() const { return vertex_clustering; }

    size_t GetNVertices() const { return vertices.Size(); }
    Vertex* GetVertex(size_t index) const { return vertices[index].get(); }

    size_t GetNTriangles() const { return triangles.Size(); }
    const Triangle& GetTriangle(size_t index) const  { return *triangles[index]; }
    Array<size_t>& GetBoundaries() { return boundaries; }
    double GetVoxelSize() const { return m; }
    shared_ptr<VoxelData> GetVoxelData() { return data; }

  protected:
    void PartitionVertices();
  private:
    inline bool isUsedVoxel(size_t x, size_t y, size_t z)
    {
      SPDLOG_DEBUG(log, "Check voxel (" + to_string(x) + "," + to_string(y) + "," + to_string(z) + ")");
      if (boundaries.Size())
        {
          if (x < boundaries[0] || x > boundaries[1] || y < boundaries[2] || y > boundaries[3]
              || z < boundaries[4] || z > boundaries[5]){
            SPDLOG_DEBUG(log, "Out of boundaries");
            return false;
          }
        }
      SPDLOG_DEBUG(log, "In boundaries, material = " + to_string((*data)(x,y,z)));
      if (!materials.Size())
        return (*data)(x,y,z);
      if (materials.Contains((*data)(x,y,z)))
        return true;
      return false;
    }
    void GenerateTrianglesAndVertices();
    void GenerateTVCube(size_t x, size_t y, size_t z,
                        Array<unique_ptr<Vertex>>& thread_vertices);
    Vertex* MakeVertex(Array<unique_ptr<Vertex>>& thread_vertices,
                       size_t x1, size_t y1, size_t z1, size_t x2, size_t y2, size_t z2);

  public:
    struct Vertex{
      //int x1,y1,z1;
      Vec<3,double> x; //INNER POINT
      //int x2,y2,z2;
      Vec<3,double> y; //OUTER POINT
      Vec<3,double> xy;
      double ratio;
      int nn = 0; //number of neighbours
      Array<Triangle*> neighbours;
      // cluster of vertex coloring
      size_t cluster;
      // vertex patch of vertex
      std::set<Vertex*> patch;

      Vertex(Vec<3,double> ax, Vec<3,double> ay) : x(ax), y(ay){
        SetRatio(0.5);
        patch.insert(this);
      }
      Vertex(Vertex* v1, Vertex* v2)
      {
        x = 0.5*(v1->x+v2->x);
        y = 0.5*(v1->y+v2->y);
        SetRatio(0.5*(v1->ratio+v2->ratio));
        patch.insert(this);
      }
      Vertex() {
        patch.insert(this);
      }
      Vertex(const Vertex&) = delete;

      inline bool doIhave(Vec<3,double> &x, Vec<3,double> &y) {
        return ((x==this->x && y==this->y) || (x==this->y && y==this->x)); }

      inline void SetRatio(double r)
      {
        ratio = r;
        xy = (1-ratio)*x + ratio*y;
        for (int i = 0; i<nn; i++)
          neighbours[i]->UpdateNormal();
      }
      inline void addNeighbour(Triangle* tri)
      {
        neighbours.Append(tri);
        nn++;
        for (auto v : {tri->v1,tri->v2,tri->v3})
          patch.insert(v);
      }
      inline void DeleteNeighbour(Triangle* tri)
      {
        for(auto i : Range(nn))
          {
            if(neighbours[i] == tri)
              {
                neighbours.DeleteElement(i);
                nn--;
                for(auto v : {tri->OtherVertices(this,0), tri->OtherVertices(this,1)})
                  {
                    bool isinother = false;
                    for(auto tri : this->neighbours)
                      if(tri->doIhave(v))
                        {
                          isinother = true;
                          break;
                        }
                    if(!isinother)
                      patch.erase(v);
                  }
                return;
              }
          }
        throw Exception("Wanted to delete non existing neighbor!");
      }

      inline void FindFreeCluster()
      {
        auto cluster_free = [&](size_t cluster)
          {
            for(auto v : patch)
              for(auto v2 : v->patch)
                if(v2 != this && v2->cluster == cluster)
                  return false;
            return true;
          };
        size_t clus = 0;
        while (!cluster_free(clus))
          clus++;
        // SPDLOG_DEBUG(log, "Vertex " + to_string() + " changed to cluster " + std::to_string(cluster));
        this->cluster = clus;
      }

      inline string to_string() {
        return "(" + std::to_string(x[0]) + "," + std::to_string(x[1]) + "," +
          std::to_string(x[2]) + ") -> (" + std::to_string(y[0]) + "," +
          std::to_string(y[1]) + "," + std::to_string(y[2]) + ")";
      }
    };

    struct Triangle{
      Vertex* v1 = nullptr;
      Vertex* v2 = nullptr;
      Vertex* v3 = nullptr;
      Vec<3,double> n;
      double area;
      int subdivisionParameter=0; //0==no subdivision, 1==cut in half because of neigbours, 2==cut in 4

      Triangle() { }
      Triangle(Vertex* v1, Vertex* v2, Vertex* v3) : v1(v1), v2(v2), v3(v3){
	Vec<3,double> temp1 = v2->xy-v1->xy;
	Vec<3,double> temp2 = v3->xy-v1->xy;
	n = Cross(temp1,temp2);
	area = L2Norm(n); //double area...
	if (area == 0) cout << "area = 0 in new triangle without bool" << endl;
	n = 1/area*n;
	//swap v2 & v3 if n is wrong orientated:
	Vec<3,double> pv = 1./3*(v1->y-v1->xy + v2->y-v2->xy + v3->y-v3->xy);
	if (InnerProduct(pv,n) < 0){
          this->v2 = v3;
          this->v3 = v2;
          n = (-1)*n;
	}
	area /= 2;
      }

      Triangle(Vertex* v1, Vertex* v2, Vertex* v3, bool updateNormal):v1(v1), v2(v2), v3(v3)
      {
	this->UpdateNormal();
      }

      Triangle(const Triangle&) = delete;

      inline void UpdateNormal()
      {
	Vec<3,double> temp1 = v2->xy-v1->xy;
	Vec<3,double> temp2 = v3->xy-v1->xy;
	n = Cross(temp1,temp2);
	area = L2Norm(n); //double area...
	if (area == 0) cout << "area = 0 in updateNormal" << endl;
	n = 1/area*n;
	area /= 2;
      }
      inline Vertex* OtherVertices(Vertex* v, int i)
      {
	if (v1==v) {if (i==1) return v2; else return v3;}
	if (v2==v) {if (i==1) return v3; else return v1;}
	if (v3==v) {if (i==1) return v1; else return v2;}
        throw Exception("No other vertices! Vertex not in triangle!");
      }

      inline bool doIhave(Vertex* v){
	if (v1==v) return true;
	if (v2==v) return true;
	if (v3==v) return true;
	return false;
      }

      Triangle* NeighbourTriangle(Vertex* v){
	Vertex* e1 = OtherVertices(v, 1);
	Vertex* e2 = OtherVertices(v, 2);
	for (int i = 0; i < e1->nn; i++){
          if (e1->neighbours[i]->doIhave(e2) && !(e1->neighbours[i]->doIhave(v)))
            {
              return e1->neighbours[i];
            }
	}
	return nullptr;
      }
      int SetSubdivisionParameter(){
	int sum = 0;
	if(this->subdivisionParameter == 2) return 0; // return if already set to 2 to avoid multiple neighbour division
	if(this->subdivisionParameter == 0) sum+=3; //only count new triangles
	if(this->subdivisionParameter == 1) sum+=2;
	this->subdivisionParameter = 2;
	sum+= this->NeighbourTriangle(v1)->SubdivisionNeighbour();
	sum+= this->NeighbourTriangle(v2)->SubdivisionNeighbour();
	sum+= this->NeighbourTriangle(v3)->SubdivisionNeighbour();
	return sum;
      }

      int SubdivisionNeighbour(){
	if (subdivisionParameter==0){
          subdivisionParameter++;
          return 1;
	}
	if (subdivisionParameter == 1){
          return this->SetSubdivisionParameter();
	}
	if (subdivisionParameter == 2){
          return 0;
	}
      }

      void MakeSubdivisionVertex(Array<Vertex*>& openVertices, Array<Vertex*>& vertex,
                                 Array<unique_ptr<Vertex>>& newVertices,
                                 v2v_map& vox_to_vert);
      inline string to_string()
      {
        return string("Trig \n") + v1->to_string() + "\n" + v2->to_string() + "\n" + v3->to_string();
      }
    };

  };

}

#endif // __FILE_GEOMETRY_HPP
