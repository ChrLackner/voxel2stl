# ifndef __FILE_SMOOTHER_HPP
# define __FILE_SMOOTHER_HPP

namespace voxel2stl
{
  class Smoother : public pyspdlog::LoggedClass
  {
  protected:
    shared_ptr<VoxelSTLGeometry> geo;
  public:
    Smoother(shared_ptr<VoxelSTLGeometry> _geo, shared_ptr<spdlog::logger> log)
      : geo(_geo), pyspdlog::LoggedClass(log) { ; }

    virtual void Apply() = 0;
  };

  class NewtonSmoother : public Smoother
  {
  public:
    NewtonSmoother(shared_ptr<VoxelSTLGeometry> geo, shared_ptr<spdlog::logger> log)
      : Smoother(geo,log) { ; }

    virtual void Apply();

  protected:
    virtual double Energy();
    double Newton(std::function<double (double)>  func, double r0,
                  double &energydifference, double & energy);
  };

  class FirstSmoother : public NewtonSmoother
  {
    double weight_area, weight_minimum;
  public:
    FirstSmoother(shared_ptr<VoxelSTLGeometry> geo,
                  double wa, double wm, shared_ptr<spdlog::logger> log)
      : NewtonSmoother(geo,log), weight_area(wa), weight_minimum(wm) { ; }

    virtual void Apply();
  };

}

# endif // __FILE_SMOOTHER_HPP
