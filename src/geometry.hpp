#ifndef __FILE_GEOMETRY_HPP
#define __FILE_GEOMETRY_HPP

namespace voxel2stl
{
  class VoxelSTLGeometry : public pyspdlog::LoggedClass
  {
  private:
    struct Vertex;
    struct Triangle;
    shared_ptr<VoxelData> data;
    Array<unique_ptr<Vertex>> vertices;
    Array<unique_ptr<Triangle>> triangles;
    double m;
    int num_threads;
    Array<unique_ptr<Array<Vertex*>>> vertex_clustering;
    shared_ptr<Array<int>> materials, boundaries;
    std::map<std::tuple<size_t,size_t,size_t,size_t,size_t,size_t>,Vertex*> voxel_to_vertex;

  public:
    VoxelSTLGeometry(shared_ptr<VoxelData> adata, shared_ptr<Array<int>> amaterials, shared_ptr<Array<int>> aboundaries, shared_ptr<spdlog::logger> log);

    void ApplySmoothingStep(bool subdivision, double weight_area, double weight_minimum);
    void WriteSTL(string filename);

  private:
    inline bool isUsedVoxel(int x, int y, int z)
    {
      if (boundaries)
        {
          if (x < (*boundaries)[0] || x > (*boundaries)[1] || y < (*boundaries)[2] || y > (*boundaries)[3]
              || z < (*boundaries)[4] || z > (*boundaries)[5])
            return false;
        }
      if (!materials)
        return (*data)(x,y,z);
      if (materials->Contains((*data)(x,y,z)))
        return true;
      return false;
    }
    void GenerateTrianglesAndVertices();
    void PartitionVertices();
    void SubdivisionTriangles();
    void MinimizeEnergy(double weight_area, double weight_minimum);
    void GenerateTVCube(int x, int y, int z,
                        Array<unique_ptr<Vertex>>& thread_vertices);
    Vertex* MakeVertex(Array<Vertex*>& local_vertices,
                       Array<unique_ptr<Vertex>>& thread_vertices,
                       int x1, int y1, int z1, int x2, int y2, int z2);

    struct Vertex{
      //int x1,y1,z1;
      Vec<3,double> x; //INNER POINT
      //int x2,y2,z2;
      Vec<3,double> y; //OUTER POINT
      Vec<3,double> xy;
      double ratio;
      int nn = 0; //number of neighbours
      Array<Triangle*> neighbours;

      Vertex(Vec<3,double> ax, Vec<3,double> ay) : x(ax), y(ay){
        SetRatio(0.5); }
      Vertex(Vertex* v1, Vertex* v2)
      {
        x = 0.5*(v1->x+v2->x);
        y = 0.5*(v1->y+v2->y);
        SetRatio(0.5*(v1->ratio+v2->ratio));
      }
      Vertex() { ; }
      inline bool doIhave(Vec<3,double> &x, Vec<3,double> &y) {
        return ((x==this->x && y==this->y) || (x==this->y && y==this->x)); }

      inline void SetRatio(double r)
      {
        ratio = r;
        xy = (1-ratio)*x + ratio*y;
        for (int i = 0; i<nn; i++)
          neighbours[i]->UpdateNormal();
      }
      inline int getx1() { return x[0]; }
      inline void addNeighbour(Triangle* tri)
      {
        neighbours.Append(tri);
        nn++;
      }
      inline void DeleteNeighbour(Triangle* tri)
      {
        for(auto i : Range(nn))
          {
            if(neighbours[i] == tri)
              {
                neighbours.DeleteElement(i);
                nn--;
                return;
              }
          }
        throw Exception("Wanted to delete non existing neighbor!");
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
                                 std::map<std::tuple<size_t,size_t,size_t,size_t,size_t,size_t>,Vertex*>& vox_to_vert);
    };

  };

  double Newton(std::function<double (double)>  func, double r0, double &energydifference, double & energy);


}

#endif // __FILE_GEOMETRY_HPP
