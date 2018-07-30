
#ifdef TEST_GEOMETRY

namespace voxel2stl
{
  class TestGeometry : public VoxelSTLGeometry
  {
  private:
    Array<unique_ptr<Vertex>> boundary_vertices;
  public:
    TestGeometry(Array<double>& avertices, Array<size_t>& atrigs,
                 shared_ptr<spdlog::logger> log)
      : VoxelSTLGeometry(nullptr,
                         Array<size_t>(),
                         Array<size_t>(),
                         log)
    {
      m = 1;
      Array<unique_ptr<Vertex>> all_vertices;
      for(auto i : Range(avertices.Size()/3))
        all_vertices.Append(make_unique<Vertex>(Vec<3,double>(int(avertices[(i*3)]),
                                                              int(avertices[(i*3)+1]),
                                                              int(avertices[(i*3)+2])),
                                                Vec<3,double>(int(avertices[(i*3)])+1,
                                                              int(avertices[(i*3)+1])+1,
                                                              int(avertices[(i*3)+2])+1)));
      for(auto i : Range(atrigs.Size()/3))
        {
          auto trig = make_unique<Triangle>(&*all_vertices[atrigs[(i*3)]],
                                            &*all_vertices[atrigs[(i*3)+1]],
                                            &*all_vertices[atrigs[(i*3)+2]]);
          for (auto j : Range(3))
            all_vertices[atrigs[(i*3)+j]]->addNeighbour(trig.get());
          triangles.Append(move(trig));
        }

      for(auto i : Range(triangles.Size()))
        {
          auto& trig = *triangles[i];
          auto hasNeighbour = [&](Vertex* v1, Vertex* v2)
            {
              for(auto j : Range(triangles.Size()))
                {
                  if(j==i)
                    continue;
                  auto& other = *triangles[j];
                  if(other.doIhave(v1) && other.doIhave(v2))
                    return true;
                }
              return false;
            };
          auto removeIfHasNoNeighbour = [&](Vertex* v1, Vertex*v2, int i1, int i2)
            {
              if(!hasNeighbour(v1,v2))
                {
                  if(all_vertices[atrigs[(i*3)+i1]])
                    {
                      boundary_vertices.Append(move(all_vertices[atrigs[(i*3)+i1]]));
                      all_vertices[atrigs[(i*3)+i1]] = nullptr;
                    }
                  if(all_vertices[atrigs[(i*3)+i2]])
                    {
                      boundary_vertices.Append(move(all_vertices[atrigs[(i*3)+i2]]));
                      all_vertices[atrigs[(i*3)+i2]] = nullptr;
                    }
                }
            };
          removeIfHasNoNeighbour(trig.v1,trig.v2,0,1);
          removeIfHasNoNeighbour(trig.v2,trig.v3,1,2);
          removeIfHasNoNeighbour(trig.v1,trig.v3,0,2);
        }
      for(auto i : Range(all_vertices.Size()))
        if(all_vertices[i])
            vertices.Append(move(all_vertices[i]));
      for(auto i : Range(vertices.Size()))
        vertices[i]->FindFreeCluster();
      PartitionVertices();
      cout << "size of boundary vertices = " << boundary_vertices.Size() << endl;
      cout << "size of vertices = " << vertices.Size() << endl;
    }
  };
}

#endif // TEST_GEOMETRY
