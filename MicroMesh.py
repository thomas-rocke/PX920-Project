'''
Microscale mesh solving


'''


from Meshing import Mesh
from FEMSolver import FEM
import numpy as np



class MicroMesh(Mesh):
  def __init__(self, nx, ny):
    corners = np.array([[0, 1],
                        [1, 1],
                        [1, 0],
                        [0, 0]])
    super().__init__(corners, nx, ny)

    self.fixed_corners = np.array([0, nx-1, nx*(ny-1), nx * ny - 1])

    self.vert_slave = nx * np.array(range(ny))[1:-1]
    self.hor_slave = np.array(range(nx))[1:-1]

    self.vert_master = self.vert_slave + (nx - 1)
    self.hor_master = self.hor_slave + nx*(ny-1)






mesh = MicroMesh(10, 12)
print(mesh.vert_slave, mesh.hor_slave)
print(mesh.vert_master, mesh.hor_master)

mesh.plot()