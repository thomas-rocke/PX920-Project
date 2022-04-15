'''
Microscale mesh solving


'''


from ast import Lambda
from cProfile import label
from turtle import fillcolor
from Meshing import Mesh, Element
from FEMSolver import FEM, gauss_eval_points, gauss_weights, J, Plane_Strain, Plane_Stress
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def random_microstructure(els, vol_frac, E1, nu1, **kwargs):
  el_IDs = np.array([i for i in range(els.shape[0])])
  mat_1_elements = np.random.choice(el_IDs, int(vol_frac * els.shape[0]), replace=False)

  for el in els[mat_1_elements]:
    el.E = E1
    el.nu = nu1
  return els


def circles_microstructure(els, vol_frac, E1, nu1, num_circ=1, offset=1/12, **kwargs):
  area_per_circ = vol_frac / num_circ
  circ_rad = np.sqrt(area_per_circ / np.pi) # Radius of each circle in microstructure
  circ_per_side = int(np.sqrt(num_circ))

  coords = np.linspace(0, 1, 2*circ_per_side + 1)[1::2]
  center = np.zeros((2))
  COMs = np.array([el.COM for el in els])
  for i in range(circ_per_side):
    for j in range(circ_per_side):
      center = np.array([coords[i], coords[j]])
      center += np.random.rand(2) * (circ_rad * 2 * offset) - (circ_rad * offset)
      mat_1_elements = np.where(np.sum((COMs - center)**2, axis=1) < circ_rad**2)
      for el in els[mat_1_elements]:
        el.E = E1
        el.nu = nu1
  return els





class MicroMesh(Mesh):
  def __init__(self, nx:int, ny:int, E1:float, E2:float, nu1:float, nu2:float, percent2:float, micro_fun=random_microstructure, **kwargs):
    corners = np.array([[0, 1],
                        [1, 1],
                        [1, 0],
                        [0, 0]])
    super().__init__(corners, nx, ny, E2, nu2)

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

    self.ELS = micro_fun(self.ELS, self.frac2, E1, nu1, **kwargs)

  def plot(self):
    '''
    Plots the constructed mesh
    '''
    title = "Mesh"

    for el in self.ELS:
      if (el.E == self.E1):
        color = 'C1'
      else:
        color = 'C2'
      plt.fill(el.XY[:, 0], el.XY[:, 1], edgecolor='k', color=color, alpha=0.3)


    # if show_ids:
    #     for i in range(4):                             #loop over all nodes within an element
    #         for el in self.ELS:                  #loop over all elements
    #             sh=0.01
    #             plt.text(el.XY[i, 0]+sh,el.XY[i, 1]+sh, el.nodes[i])

    # Set chart title.
    plt.title(title, fontsize=19)
    # Set x axis label.
    plt.xlabel("$x_1$", fontsize=10)
    # Set y axis label.
    plt.ylabel("$x_2$", fontsize=10)

    #plt.legend()

    plt.show()


    

class MicroSolver(FEM):
  @property
  def T(self):
    return self.mesh.T

  def Voigt(self):
    C_1 = self.elasticity(self.mesh.E1, self.mesh.nu1)
    C_2 = self.elasticity(self.mesh.E2, self.mesh.nu2)

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
    for e in tqdm(range(len(self.mesh.ELS)), desc="Calculating elemental Rs", leave=False):
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
    for e in tqdm(range(len(self.mesh.ELS)), desc=f"Calculating elemental σs", leave=False):
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

    for i in tqdm(range(3), desc="Testing under Unit Strains", leave=False):
      eps = epss[i]
      self.mesh.forces = self.r_vec(eps*mag)
      self.solve()
      C[:, i] = self.sigma(eps*mag)/mag

      self.C = C
    return C

  def infer_props(self, C=None):
    if C is None:
      C = self.C
    C_11 = 0.5*(C[0, 0] + C[1, 1])
    C_12 = 0.5*(C[0, 1] + C[1, 0])

    if self.elasticity == Plane_Stress:
      nu = C_12/C_11
      E = C_11 * (1 - nu * nu)

    elif self.elasticity == Plane_Strain:
      nu = C_12 / (C_11 + C_12)
      E  = C_12 * ((1-2*nu)*(1+nu))/nu

    else:
      print("Process to invert C unknown")
      nu = 0
      E = 0
    
    return E, nu




        






# num_circ = 4

# mat_1_params=[10E9, 0.32]
# mat_2_params=[80E9, 0.22]
# rel_conc=0.55

# E_1, nu_1 = mat_1_params
# E_2, nu_2 = mat_2_params


# mesh = MicroMesh(96, 96, E_1, E_2, nu_1, nu_2, rel_conc, micro_fun=circles_microstructure)
# mesh.plot()
# solver = MicroSolver(mesh)
# print(solver.homogenize())