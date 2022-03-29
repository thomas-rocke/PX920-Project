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

    self.all_slaves = np.sort(np.append(self.vert_slave, self.hor_slave))
    self.slave_DOFs = np.sort(np.append(2 * self.all_slaves, 2*self.all_slaves + 1))

    self.all_IDs = np.array(range(nx * ny))
    self.free_nodes = np.sort(np.array([ID for ID in self.all_IDs if ID not in self.all_slaves]))
    self.free_DOFs = np.sort(np.append(2 * self.free_nodes, 2 * self.free_nodes + 1))

    self.all_masters = np.sort(np.append(self.vert_master, self.hor_master))
    self.master_DOFs = np.sort(np.append(2*self.all_masters, 2*self.all_masters + 1))

    self.nnodes = nx * ny
    self.num_slaves = (nx + ny - 4)
    self.num_masters = self.num_slaves

    G = np.zeros((2*self.nnodes, 2*self.nnodes))

    # Master node DOF
    G[np.ix_(2*self.vert_slave, 2*self.vert_master)] = 1.0
    G[np.ix_(2*self.hor_slave, 2*self.hor_master)] = 1.0
    G[np.ix_(2*self.vert_slave + 1, 2*self.vert_master + 1)] = 1.0
    G[np.ix_(2*self.hor_slave + 1, 2*self.hor_master + 1)] = 1.0

    # Slave node DOF
    G[2*self.vert_slave, 2*self.vert_slave] = -1
    G[2*self.hor_slave, 2*self.hor_slave] = -1
    G[2*self.vert_slave + 1, 2*self.vert_slave + 1] = -1
    G[2*self.hor_slave + 1, 2*self.hor_slave + 1] = -1
    
    self.Gs = G[np.ix_(self.slave_DOFs, self.slave_DOFs)]
    
    self.Gm = G[np.ix_(self.slave_DOFs, self.master_DOFs)]

    self.T = np.zeros((2*self.nnodes, 2*self.num_masters))
    self.T[self.master_DOFs, :] = np.identity(2*self.num_masters)
    self.T[self.slave_DOFs, :] = -1 * np.linalg.inv(self.Gs) @ self.Gm

    





mesh = MicroMesh(4, 4)
for el in mesh.ELS:
  print(el.XY)
mesh.plot()