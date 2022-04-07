'''
Solves FEM given mesh and material properties

'''
from cmath import pi
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import griddata
from Meshing import Mesh


gauss_eval_points = { # Evaluation coordinates for Gaussian Quadrature integration
    1 : np.array([0]),
    2 : np.array([-np.sqrt(1/3), np.sqrt(1/3)]),
    3 : np.array([-np.sqrt(3/5), 0, np.sqrt(3/5)]),
    4 : np.array([-np.sqrt(3/7 + 2 * np.sqrt(6/5)/7), -np.sqrt(3/7 - 2 * np.sqrt(6/5)/7), np.sqrt(3/7 - 2 * np.sqrt(6/5)/7), np.sqrt(3/7 + 2 * np.sqrt(6/5)/7)]),
    5 : np.array([-np.sqrt(5 + 2 * np.sqrt(10/7))/3, -np.sqrt(5 - 2 * np.sqrt(10/7))/3, 0, np.sqrt(5 - 2 * np.sqrt(10/7))/3, np.sqrt(5 + 2 * np.sqrt(10/7))/3])
}

gauss_weights = { # Weights for Gaussian Quadrature integration
    1 : np.array([2]),
    2 : np.array([1, 1]),
    3 : np.array([5/9, 8/9, 5/9]),
    4 : np.array([(18 - np.sqrt(30))/36, (18 + np.sqrt(30))/36, (18 + np.sqrt(30))/36, (18 - np.sqrt(30))/36]),
    5 : np.array([(322 - 13 * np.sqrt(70))/900, (322 + 13 * np.sqrt(70))/900, 128/225, (322 + 13 * np.sqrt(70))/900, (322 - 13 * np.sqrt(70))/900])
}


def Plane_Stress(E, nu):
    '''C matrix in the case of plane stress'''
    const = E/(1.0-nu*nu)
    C = np.array([[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5*(1.0-nu)]])
    return const*C

def Plane_Strain(E, nu):
    '''C matrix in the case of plane strain'''
    const = E/(1.0-nu)/(1 + nu)
    C = np.array([[1.0 - nu, nu, 0.0], [nu, 1.0-nu, 0.0], [0.0, 0.0, 0.5*(1.0-2 * nu)]])
    return const*C


def N(xi, eta):
    '''Nodal functions; returns a vector as a function of xi, eta'''
    N1 = 0.25*(1.0-xi)*(1.0+eta)
    N2 = 0.25*(1.0-xi)*(1.0-eta)
    N3 = 0.25*(1.0+xi)*(1.0-eta)
    N4 = 0.25*(1.0+xi)*(1.0+eta)
    return np.array([N1, N2, N3, N4])


def J(xi, eta):
    '''Nodal Jacobian as a function of xi, eta'''
    return np.array([[1.0, 0.0],[0.125 - 0.125*eta, 0.375 - xi*0.125]])

def J_inv(xi, eta):
    '''Inverse Nodal Jacobian'''
    Jac = J(xi, eta)
    return np.linalg.inv(Jac)


class FEM():
    def __init__(self, mesh:Mesh, elasticity_func = Plane_Strain, quad_points=2):
        '''
        Initialise FEM solver
        '''

        self.mesh = mesh 
        self.elasticity = elasticity_func
        self.quad_points = quad_points

        self.nnodes = self.mesh.XY.shape[0]
        self.K = np.zeros((2 * self.nnodes, 2 * self.nnodes))
        self.displacements = np.zeros((2 * self.nnodes))
        self.Stresses = None
        self.Strains = None

    @property
    def forces(self):
        '''
        Alias for forces
        '''
        return self.mesh.forces


    def strain_displacement(self, corners, xi, eta):
        '''
        Find the strain-displacement matrix for the element given by corners, xi, eta
        '''

        dN = self.dN(xi, eta)

        # Real-space Jacobian
        Jmat = dN @ corners
        Jmat_inv = np.linalg.inv(Jmat)

        dNdx = np.dot(Jmat_inv, dN)

        # Strain-displacement matrix
        B = np.zeros((3,8))
        B[0,0::2] = dNdx[0,:]
        B[1,1::2] = dNdx[1,:]
        B[2,0::2] = dNdx[1,:]
        B[2,1::2] = dNdx[0,:]
        return B

    def dN(self, xi, eta):
        nat_coords = np.array([[-1, 1, 1, -1],[-1, -1, 1, 1]]) # Natural (square) coord system
        
        dN = np.zeros((2,4)) # Gradient of Shape functions
        dN[0,:]=(1/4)*nat_coords[0,:]*(1+nat_coords[1,:]*eta)
        dN[1,:]=(1/4)*nat_coords[1,:]*(1+nat_coords[0,:]*xi)

        return dN
    def Gauss_quad(self, corners, E, nu):
        '''
        Perform Gauss Quadrature integration using self.quad_points**2 total points
        '''
        points = gauss_eval_points[self.quad_points]
        weights = gauss_weights[self.quad_points]
        k_element = np.zeros((8, 8))

        C = self.elasticity(E, nu)
        for i in range(self.quad_points):
            for j in range(self.quad_points):
                xi = points[i]
                eta = points[j]

                # Strain-displacement matrix
                B = self.strain_displacement(corners, xi, eta)

                # Construct elemental k
                k_element += (B.T @ C @ B) * weights[i] * weights[j]
        return k_element     

    def eval_K(self):
        '''
        Evaluate stiffness matrix for all integration points
        '''
        # Reset K to 0
        self.K = np.zeros((2 * self.nnodes, 2 * self.nnodes))
        for i in tqdm(range(self.mesh.ELS.shape[0]), desc="Calculating elemental Ks"):
            element = self.mesh.ELS[i]
            # Find local k matrix
            corners = element.XY
            k_element = self.Gauss_quad(corners, element.E, element.nu)

            # Add local matrices together
            self.K[np.ix_(element.DOF, element.DOF)] += k_element

    def solve(self):
        '''
        Solves the Homogeneous FEM problem given by the mesh, pinning, loading, and material properties
        '''
        # Compute K matrix
        self.eval_K()
        # Separate the K matrix out into components
        is_pinned = self.mesh.pins.flatten()
        K_ee = self.K[is_pinned == False, :][:, is_pinned == False]
        K_ef = self.K[is_pinned == False, :][:, is_pinned == True]
        K_ff = self.K[is_pinned == True, :][:, is_pinned == True]

        Forces = self.mesh.forces.flatten()[is_pinned == False]

        disps = np.linalg.solve(K_ee, Forces)

        reactions = K_ef.T @ disps
        self.displacements[is_pinned==False] = disps
        self.total_force = (self.K @ self.displacements)

    def show_deformation(self, magnification=1):

        disps = -magnification * self.displacements

        old_XY = self.mesh.XY
        new_XY = self.mesh.XY + disps.reshape((self.nnodes, 2))
        plt.plot(old_XY[:, 0], old_XY[:, 1], 'sk', label='Undeformed shape')
        plt.plot(new_XY[:, 0], new_XY[:, 1], 'sr', label='Deformed Shape')

        for el in self.mesh.ELS:
            plt.fill(old_XY[el.nodes, 0], old_XY[el.nodes, 1], edgecolor='k', fill=False)
            plt.fill(new_XY[el.nodes, 0], new_XY[el.nodes, 1], edgecolor='r', fill=False)
        # Set chart title.
        plt.title("Mesh Deformation under loading", fontsize=19)
        # Set x axis label.
        plt.xlabel("$x_1$", fontsize=10)
        # Set y axis label.
        plt.ylabel("$x_2$", fontsize=10)

        plt.legend()

        plt.show()


