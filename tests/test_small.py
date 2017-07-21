
from voxel2stl import *
from pyspdlog import *
from struct import pack
import stl
from matplotlib import pyplot
from mpl_toolkits import mplot3d

def CreateVoxel(filename,indicator_function,dimensions):
    l = []
    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            for z in range(dimensions[2]):
                l.append(indicator_function(x,y,z))
    with open(filename,"wb") as file:
        file.write(pack(str(dimensions[0]*dimensions[1]*dimensions[2])+'B',*l))

def test_small():
    sinkfile = simple_file_sink("debug.out",True)
    sinkconsole = console_sink(True)
    logger = Logger("Geometry",[sinkfile,sinkconsole])
    logger.SetLevel(LOGGING_LEVEL.DEBUG)
    logger.SetPattern("%t %+")
    sinkconsole.SetLevel(LOGGING_LEVEL.INFO)
    sinkfile.SetLevel(LOGGING_LEVEL.DEBUG)
    CreateVoxel("voxel_small.raw",
                lambda x,y,z: int(x>3) * int(x<7) * int(y>3) * int(y<7) * int(z>3) * int(z<7),
                (10,10,10))
    data = VoxelData("voxel_small.raw", 10,10,10, 1, log=logger)
    geo = VoxelSTLGeometry(data,logger=logger)
    # geo.ApplySmoothingStep(False)
    geo.WriteSTL("small.stl")
    stl_mesh = stl.mesh.Mesh.from_file("small.stl")
    if __name__ == "__main__":
        figure = pyplot.figure()
        axes = mplot3d.Axes3D(figure)
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(stl_mesh.vectors))
        scale = stl_mesh.points.flatten(-1)
        axes.auto_scale_xyz(scale,scale,scale)
        pyplot.show()
    volume, cog, inertia = stl_mesh.get_mass_properties()
    print("Volume: ", volume)
    print("Cog: ", cog)
    print("Inertia: ", inertia)

if __name__ == "__main__":
    test_small()
