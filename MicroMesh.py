'''
Microscale mesh solving


'''


from cProfile import label
from Meshing import Mesh, Element
from FEMSolver import FEM, gauss_eval_points, gauss_weights, J
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class MicroMesh(Mesh):
  def __init__(self, nx:int, ny:int, E1:float, E2:float, nu1:float, nu2:float, percent2:float):
    corners = np.array([[0, 1],
                        [1, 1],
                        [1, 0],
                        [0, 0]])
    super().__init__(corners, nx, ny, E1, nu1)

    self.E1 = E1
    self.E2 = E2
    self.nu1 = nu1
    self.nu2 = nu2
    self.frac2 = percent2
    self.frac1 = 1 - percent2

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

  def Voigt(self):
    C_1 = self.elasticity(self.mesh.E1, self.mesh.nu1)
    C_2 = self.elasticity(self.mesh.E2, self.mesh.nu2)

    print(self.mesh.nu1)

    return self.mesh.frac1 * C_1 + self.mesh.frac2 * C_2

  def Reuss(self):
    C_1 = self.elasticity(self.mesh.E1, self.mesh.nu1)
    C_2 = self.elasticity(self.mesh.E2, self.mesh.nu2)

    return 1/(self.mesh.frac1 / C_1 + self.mesh.frac2 / C_2)


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

  def r_vec(self, eps):
    r_vec = np.zeros(2*self.mesh.nnodes)

    points = gauss_eval_points[self.quad_points]
    weights = gauss_weights[self.quad_points]
    for e in tqdm(range(len(self.mesh.ELS)), desc="Calculating elemental Rs"):
      element = self.mesh.ELS[e]
      r_vec_e = np.zeros((8))
      C = self.elasticity(element.E, element.nu)
      for i in range(self.quad_points):
        for j in range(self.quad_points):
          xi = points[i]
          eta = points[j]

          B = self.strain_displacement(element.XY, xi, eta)
          J = self.dN(xi, eta) @ element.XY

          r_vec_e += B.T @ C @ eps * np.linalg.det(J) * weights[i] * weights[j]
      r_vec[np.ix_(element.DOF)] += r_vec_e
    return r_vec

  def sigma(self, eps):
    disps = self.displacements
    points = gauss_eval_points[self.quad_points]
    weights = gauss_weights[self.quad_points]
    sigma = np.zeros((3))
    vol = 0
    for e in tqdm(range(len(self.mesh.ELS)), desc=f"Calculating elemental σs"):
      element = self.mesh.ELS[e]
      C = self.elasticity(element.E, element.nu)
      d = disps[element.DOF]
      for i in range(self.quad_points):
        for j in range(self.quad_points):
          xi = points[i]
          eta = points[j]
          B = self.strain_displacement(element.XY, xi, eta)
          J = self.dN(xi, eta) @ element.XY

          sigma += C @ (B @ d + eps) * np.linalg.det(J) * weights[i] * weights[j]
          vol += np.linalg.det(J)
    return sigma / vol

  def homogenize(self, mag=1):
    epss = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]

    C = np.zeros((3, 3))

    for i, eps in enumerate(epss):
      self.mesh.forces = self.r_vec(eps*mag)
      self.solve()
      C[:, i] = self.sigma(eps*mag)/mag
    return C



        







mesh = MicroMesh(20, 20, 10E9, 80E9, 0.32, 0.22, 0.45)
mesh.apply_load((0, 2), 'top')
#mesh.apply_load((0.5, 0), 'left')
#mesh.apply_load((-0.2, 0.1), 'right')

#mesh.plot()
solver = MicroSolver(mesh)

print(solver.homogenize(1))
#print(solver.Voigt())
#print(solver.Reuss())
#solver.solve()
#solver.show_deformation(10)