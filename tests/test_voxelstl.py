
from ngsolve import *
from voxel2stl import *
from pyspdlog import *
from struct import pack
from netgen.stl import *
from numpy import pi

def CreateVoxel(filename,indicator_function,dimensions):
    l = []
    for x in range(dimensions[0]):
        for y in range(dimensions[1]):
            for z in range(dimensions[2]):
                l.append(indicator_function(x,y,z))
    with open(filename,"wb") as file:
        file.write(pack(str(dimensions[0]*dimensions[1]*dimensions[2])+'B',*l))


def test_voxelSTL():
    sinkfile = simple_file_sink("debug.out",True)
    sinkconsole = console_sink(True)
    logger = Logger("Geometry",[sinkfile,sinkconsole])
    logger.SetLevel(LOGGING_LEVEL.DEBUG)
    sinkconsole.SetLevel(LOGGING_LEVEL.INFO)
    sinkfile.SetLevel(LOGGING_LEVEL.DEBUG)
    SetLogger("Geometry",logger)
    CreateVoxel("voxelmodel.raw",
                lambda x,y,z: int((100-((x-50)*(x-50)+(y-50)*(y-50)+(z-50)*(z-50)))>0) +
                int((100-((x-50)*(x-50)+(y-50)*(y-50)+(z-50)*(z-50)))>0) * int(z>=50),
                (100,100,100))
    data = VoxelData("voxelmodel.raw",100,100,100,0.01)
    geo = VoxelSTLGeometry(data,materials=[2],boundaries=[-10,200,50,200,-10,200])
    logger.info("STL done, start smoothing")
    geo.ApplySmoothingStep(False)
    geo.ApplySmoothingStep(False)
    for i in range(8):
        geo.ApplySmoothingStep(True)
    geo.ApplySmoothingStep(False)
    geo.ApplySmoothingStep(False)
    logger.info("Smoothing done, start writing to stl file")
    geo.WriteSTL("model.stl")
    netgen_geo = LoadSTLGeometry("model.stl")
    mesh = Mesh(netgen_geo.GenerateMesh())
    #mesh.Curve(4)
    if __name__ == "__main__":
        return mesh
    assert abs(Integrate(CoefficientFunction(1),mesh)- 4*pi/3*0.001/4) < 2e-4

if __name__ == "__main__":
    mesh = test_voxelSTL()
    integral = Integrate(CoefficientFunction(1),mesh)
    print("Integral: ", integral)
    print("Difference: ",abs(integral - 4*pi/3*0.001/4))
    Draw(mesh)
