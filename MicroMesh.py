'''
Microscale mesh solving


'''


from Meshing import Mesh
from FEMSolver import FEM
import numpy as np
import matplotlib.pyplot as plt



class MicroMesh(Mesh):
  def __init__(self, nx:int, ny:int, E1:float, E2:float, nu1:float, nu2:float, percent2:float):
    corners = np.array([[0, 1],
                        [1, 1],
                        [1, 0],
                        [0, 0]])
    super().__init__(corners, nx, ny, E1, nu1)

    self.fixed_corners = np.array([0, nx-1, nx*(ny-1), nx * ny - 1])

    self.vert_slave = nx * np.array(range(ny))[1:-1]
    self.hor_slave = np.array(range(nx))[1:-1]

    self.vert_master = self.vert_slave + (nx - 1)
    self.hor_master = self.hor_slave + nx*(ny-1)

    self.all_slaves = np.append(self.vert_slave, self.hor_slave)
    self.slave_DOFs = np.append(2 * self.all_slaves, 2*self.all_slaves + 1)

    self.all_masters = np.append(self.vert_master, self.hor_master)
    self.master_DOFs = np.append(2*self.all_masters, 2*self.all_masters + 1)

    self.all_corners = np.array([0, nx-1, (ny - 1) * nx, ny * nx - 1])
    self.corner_DOFs = np.append(2*self.all_corners, 2*self.all_corners + 1)
    self.num_corners = 4

    self.all_IDs = np.array(range(nx * ny))
    self.free_nodes = np.array([ID for ID in self.all_IDs if ID not in self.all_slaves and ID not in self.all_corners])
    self.free_DOFs = np.append(2 * self.free_nodes, 2 * self.free_nodes + 1)

    self.pins[self.all_corners, :] = True

    self.nnodes = nx * ny
    self.num_slaves = (nx + ny - 4)
    self.num_masters = self.nnodes - self.num_slaves - self.num_corners
    '''
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
    
    self.Gm = G[np.ix_(self.slave_DOFs, self.master_DOFs)]'''

    #self.Gs = -1 * np.identity(len(self.slave_DOFs))
    #self.Gm = np.zeros((len(self.slave_DOFs), len(self.free_DOFs)))
    
    #self.Gm[self.slave_DOFs, self.master_DOFs] = 1

    G = np.zeros((2*self.nnodes, 2*self.nnodes))
    G[self.slave_DOFs, self.slave_DOFs] = -1
    G[self.slave_DOFs, self.master_DOFs] = 1

    self.Gs = G[np.ix_(self.slave_DOFs, self.slave_DOFs)]
    self.Gm = G[np.ix_(self.slave_DOFs, self.free_DOFs)]
    self.T = np.zeros((2*self.nnodes, 2*self.num_masters))
    self.T[self.free_DOFs, :] = np.identity(2*self.num_masters)
    self.T[self.slave_DOFs, :] = -1 * np.linalg.inv(self.Gs) @ self.Gm


    mat_2_elements = np.random.choice(self.ELS, int(percent2 * self.nnodes), replace=False)

    for el in mat_2_elements:
      el.E = E2
      el.nu = nu2

    

class MicroSolver(FEM):
  @property
  def T(self):
    return self.mesh.T
  
  def solve(self):
    self.eval_K()
    K_m = (self.T.T @ self.K) @ self.T
    
    is_pinned = self.mesh.pins.flatten()

    masters = self.mesh.free_DOFs
    K_ee = K_m[is_pinned[masters] == False, :][:, is_pinned[masters] == False]

    Forces = (self.T.T @ self.mesh.forces.flatten())[is_pinned[masters] == False]

    d_m = np.linalg.solve(K_ee, Forces)
    self.displacements = self.T @ d_m

    self.dm = d_m
    self.ds = self.displacements[self.mesh.slave_DOFs]
  
  def show_periodic_deformation(self, magnification=1):
    disps = -magnification * self.displacements

    old_XY = self.mesh.XY
    new_XY = self.mesh.XY + disps.reshape((self.nnodes, 2))
    plt.plot(old_XY[:, 0], old_XY[:, 1], 'sk', label='Undeformed shape')
    plt.plot(new_XY[:, 0], new_XY[:, 1], 'sr', label='Deformed Shape')

    for el in self.mesh.ELS:
        plt.fill(old_XY[el.nodes, 0], old_XY[el.nodes, 1], edgecolor='k', fill=False)
        plt.fill(new_XY[el.nodes, 0], new_XY[el.nodes, 1], edgecolor='r', fill=False)
    
    for neigh in [[0, 1], [0, -1], [1, 0], [-1, 0], [1, 1], [-1, -1], [1, -1], [-1, 1]]:
      old = old_XY + neigh
      new = new_XY + neigh
      for el in self.mesh.ELS:
        plt.fill(old[el.nodes, 0], old[el.nodes, 1], edgecolor='k', fill=False, alpha=0.3)
        plt.fill(new[el.nodes, 0], new[el.nodes, 1], edgecolor='r', fill=False, alpha=0.3)
    
    
    
    # Set chart title.
    plt.title("Mesh Deformation under loading", fontsize=19)
    # Set x axis label.
    plt.xlabel("$x_1$", fontsize=10)
    # Set y axis label.
    plt.ylabel("$x_2$", fontsize=10)

    plt.legend()

mesh = MicroMesh(6, 6, 10E9, 0.32, 80E8, 0.22, 0.45)
mesh.apply_load((0, 2), 'top')
#mesh.apply_load((0.5, 0), 'left')
#mesh.apply_load((-0.2, 0.1), 'right')

print(mesh.all_slaves)
print(mesh.all_masters)

#mesh.plot()
solver = MicroSolver(mesh)
solver.eval_K()
solver.solve()
solver.show_periodic_deformation(1)
plt.show()