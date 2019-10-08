from voxel2stl import *
from struct import pack
# import stl
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

# def test_stairs():
#     CreateVoxel("stairs.raw",
#                 lambda x,y,z: int(x*3<z) * int(x>2) * int(x<28) * int(y>2) * int(y<18)* int(z>2) * int(z<28),
#                 (30,20,30))
#     data = VoxelData("stairs.raw", 30,20,30, 1)
#     geo = VoxelSTLGeometry(data)
#     geo.ApplySmoothingStep(False)
#     geo.ApplySmoothingStep(False)
#     for i in range(6):
#         geo.ApplySmoothingStep(True)
#     geo.ApplySmoothingStep(False)
#     geo.ApplySmoothingStep(False)
#     geo.WriteSTL("stairs.stl")
#     stl_mesh = stl.mesh.Mesh.from_file("stairs.stl")
#     if __name__ == "__main__":
#         figure = pyplot.figure()
#         axes = mplot3d.Axes3D(figure)
#         axes.add_collection3d(mplot3d.art3d.Poly3DCollection(stl_mesh.vectors))
#         scale = stl_mesh.points.flatten(-1)
#         axes.auto_scale_xyz(scale,scale,scale)
#         pyplot.show()

def test_small():
    CreateVoxel("voxel_small.raw",
                lambda x,y,z: int(x>3) * int(x<7) * int(y>3) * int(y<7) * int(z>3) * int(z<7),
                (10,10,10))
    data = VoxelData("voxel_small.raw", 10,10,10, 1)
    geo = VoxelSTLGeometry(data)
    smoother = FirstSmoother(geo)
    for i in range(10):
        smoother.Apply()
    # for i in range(3):
    #     smoother.Apply()
    geo.WriteSTL("small.stl")
    stl_mesh = stl.mesh.Mesh.from_file("small.stl")
    if __name__ == "__main__":
        import subprocess
        subprocess.call(['/home/christopher/git/install/netgen/bin/netgen','small.stl'])
    # if __name__ == "__main__":
    #     figure = pyplot.figure()
    #     axes = mplot3d.Axes3D(figure)
    #     axes.add_collection3d(mplot3d.art3d.Poly3DCollection(stl_mesh.vectors))
    #     scale = stl_mesh.points.flatten(-1)
    #     axes.auto_scale_xyz(scale,scale,scale)
    #     pyplot.show()
    # volume, cog, inertia = stl_mesh.get_mass_properties()
    # print("Volume: ", volume)
    # print("Cog: ", cog)
    # print("Inertia: ", inertia)

def test_stairs():
    CreateVoxel("voxel_stairs.raw",
                lambda x,y,z: int(x>1) * int(x<18) * int(y>1) * int(y<9) * int(z>1) * int(z<18) * int(3*x<z),
                (20,10,20))
    data = VoxelData("voxel_stairs.raw", 20,10,20, 1)
    geo = VoxelSTLGeometry(data)
    smoother2 = FirstSmoother(geo)
    smoother = SimpleSmoother(geo)
    for i in range(3):
        smoother.Apply()
    for i in range(10):
        geo.SubdivideTriangles(smoother2)
        smoother2.Apply()
    for i in range(4):
        smoother2.Apply()
    geo.WriteSTL("stairs.stl")
    # if __name__ == "__main__":
    #     figure = pyplot.figure()
    #     axes = mplot3d.Axes3D(figure)
    #     axes.add_collection3d(mplot3d.art3d.Poly3DCollection(stl_mesh.vectors))
    #     scale = stl_mesh.points.flatten(-1)
    #     axes.auto_scale_xyz(scale,scale,scale)
    #     pyplot.show()


if __name__ == "__main__":
    # test_small()
    test_stairs()
