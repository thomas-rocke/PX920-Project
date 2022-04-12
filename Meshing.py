from inspect import Parameter
import matplotlib.pyplot as plt
import numpy as np

from NodeDists import *

def shared_pairs(XY1, XY2, percent_tol=0.1):
    '''
    Finds all coincident coordinates, returning a list of indeces of 
    XY1 and XY2 that correspond to shared coords 
    
    percent tol gives a spatial tolerance at which two coordinates are deemed coincident,
    based on the minimum separation of coordinates in either XY1 or XY2

    IE: i, j coincident if r_ij < percent_tol * r_min

    '''
    from scipy.spatial.distance import pdist, cdist

    r_min = np.min([np.min(pdist(XY1)), np.min(pdist(XY2))]) # find minimum nodal separation in either subgrid
    

    dist_tol = percent_tol * r_min # Tolerance for coincidence

    # Use cdist to find distances between all nodes in XY1 to all nodes in XY2
    all_dists = cdist(XY1, XY2)

    coincidence = all_dists < dist_tol 

    # Generate masks where a node is coincident with another node
    mask1 = np.sum(coincidence, axis=-1)>0
    mask2 = np.sum(coincidence, axis=0)>0


    #Generate IDs list for all nodes in XY1 and XY2
    ids1 = np.arange(0, XY1.shape[0])
    ids2 = np.arange(0, XY2.shape[0])

    #Mask to only coincident node IDs
    return ids1[mask1], ids2[mask2]



class Element():
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu
        self.XY = np.zeros((4, 2))
    
    @property
    def Area(self):
        coords = self.XY
        # Shoelace algorithm
        mat = np.zeros((5, 2))
        mat[:-1, :] = coords
        mat[-1, :] = coords[0, :]
        area = np.abs(np.sum([np.linalg.det(mat[i:i+2, :]) for i in range(4)]) / 2)
        return area

    @property
    def COM(self):
        '''
        Gets center of mass of element
        '''
        return np.average(self.XY, axis=0)



class Mesh():
    '''
    Meshing class
    
    Uses modified code from the given PX912 mesh generation functions, 
    and is heavily inspired by the PX920 workshop 1 Mesh class 

    '''
    def __init__(self, corners, nx, ny, E=0, nu=0, coord_func=Uniform):
        '''
        Inits Mesh
        corners is a shape (4, 2) array of coords
        [0, :] is top left, [1, :] top right
        [2, :] bottom right, [3, :] bottom left

        coord_func(corners, nx, ny) is a generates the coordinates for each of the integration points
        '''
        self.E = E
        self.nu = nu
        self.corners=corners
        self.all_corners = corners
        self.nx = nx
        self.ny = ny
        self.nnodes = nx * ny
        self.node_dist = coord_func
        self.make_mesh()
        self.pins = np.array([False] * 2 * nx * ny).reshape((nx*ny, 2))
        self.forces = np.zeros_like(self.pins, dtype=float)

        self.edges = {'bottom' : np.array([ny * (i + 1) - 1 for i in range(nx)]),
                      'top' : np.array([i * (ny) for i in range(nx)]),
                      'right' : np.array([i for i in range(ny)]),
                      'left' : np.array([(nx-1)*ny + i for i in range(ny)])}

    def make_mesh(self):
        '''
        Generate mesh from corners and nx, ny
        Sets self.XY to integration point coords
        with shape (2, nx * ny)
        self.ELS is shape ((nx-1) * (ny-1), 4), and gives the 4 integration points corresponding tro each node
        self.DOF is shape ((nx-1) * (ny-1), 4) and gives DOF for each element
        '''
        self.XY = np.array(self.node_dist(self.corners, self.nx, self.ny))
        nelx = self.nx - 1
        nely = self.ny - 1
        nnodes = nelx*nely
        self.ELS = np.array([Element(self.E, self.nu) for i in range(nnodes)])#np.zeros((nnodes, 4), dtype=int)

        for j in range(nelx):
            for i in range(nely):
                self.ELS[j+i*nelx].nodes = np.array([j+i*self.nx, j+i*self.nx+1,j+(i+1)*self.nx+1, j+(i+1)*self.nx])
        
        for el in self.ELS:
            el.XY = self.XY[el.nodes, :]



        self.DOF = np.zeros((nnodes, 8), dtype=int)

        for i in range(nnodes):
            nodes = self.ELS[i].nodes
            self.ELS[i].DOF = np.array([nodes[0]*2, nodes[1]*2-1, nodes[1]*2, nodes[1]*2+1, nodes[2]*2, nodes[2]*2+1, nodes[3]*2, nodes[3]*2+1])


    def plot(self, show_ids = True):
        '''
        Plots the constructed mesh
        '''
        title = "Mesh"

        for el in self.ELS:
            plt.fill(el.XY[:, 0], el.XY[:, 1], edgecolor='k', fill=False)


        if show_ids:
            for i in range(4):                             #loop over all nodes within an element
                for el in self.ELS:                  #loop over all elements
                    sh=0.01
                    plt.text(el.XY[i, 0]+sh,el.XY[i, 1]+sh, el.nodes[i])

        # Set chart title.
        plt.title(title, fontsize=19)
        # Set x axis label.
        plt.xlabel("$x_1$", fontsize=10)
        # Set y axis label.
        plt.ylabel("$x_2$", fontsize=10)

        #plt.legend()

        plt.show()

    def pin_edge(self, edge, direction):
        '''
        Pins all nodes along a given edge, in the direction specified

        edge in ('left', 'right', 'top', 'bottom')

        direction is 0 (x_1 dirn) or 1 (x_2 dirn) 
        '''

        edge_nodes = self.edges[edge]

        self.pins[edge_nodes, direction] = True


    def apply_load(self, forces, edge):
        '''
        Applies loading forces to the system
        forces is a len 2 float array,
        and edge in ('left', 'right', 'top', 'bottom') is the edge to apply forces to
        '''
        try:
           assert len(forces) == 2
        except AssertionError:
            print(f"Force length error: input length {len(forces)}, expected 2")
            pass
        edge_nodes = self.edges[edge]

        self.forces[edge_nodes, :] = forces




