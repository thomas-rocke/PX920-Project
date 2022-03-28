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


class Mesh():
    '''
    Meshing class
    
    Uses modified code from the given PX912 mesh generation functions, 
    and is heavily inspired by the PX920 workshop 1 Mesh class 

    '''
    def __init__(self, corners, nx, ny, coord_func=Uniform):
        '''
        Inits Mesh
        corners is a shape (4, 2) array of coords
        [0, :] is top left, [1, :] top right
        [2, :] bottom right, [3, :] bottom left

        coord_func(corners, nx, ny) is a generates the coordinates for each of the integration points
        '''
        self.corners=corners
        self.all_corners = corners
        self.nx = nx
        self.ny = ny
        self.nnodes = nx * ny
        self.node_dist = coord_func
        self.make_mesh()
        self.pins = np.array([False] * 2 * nx * ny).reshape((nx*ny, 2))
        self.forces = np.zeros_like(self.pins)

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
        self.ELS = np.zeros((nnodes, 4), dtype=int)

        for i in range(nelx):
            for j in range(nely):
                self.ELS[j+i*nelx, :] = [j+i*self.nx, j+i*self.nx+1,j+(i+1)*self.nx+1, j+(i+1)*self.nx]

        self.DOF = np.zeros((nnodes, 8), dtype=int)

        for i in range(nnodes):
            self.DOF[i, :] = [self.ELS[i,0]*2, self.ELS[i,1]*2-1, self.ELS[i,1]*2, self.ELS[i,1]*2+1, self.ELS[i,2]*2, self.ELS[i,2]*2+1, self.ELS[i,3]*2, self.ELS[i,3]*2+1]


    def plot(self, show_ids = True):
        '''
        Plots the constructed mesh
        '''
        title = "Mesh"

        #Fully Pinned
        cond = self.pins[:, 0] * self.pins[:, 1]

        if sum(cond):
            plt.plot(self.XY[cond==True, 0], self.XY[cond==True, 1], 'sg', label='Fully Pinned')

        #Pinned in x1 only
        cond = self.pins[:, 0] * (1 - self.pins[:, 1])
        if sum(cond):
            plt.plot(self.XY[cond==True, 0], self.XY[cond==True, 1], 'sr', label='Pinned in x1')
        #Pinned in x2 only
        cond = (1 - self.pins[:, 0]) * self.pins[:, 1]
        if sum(cond):
            plt.plot(self.XY[cond==True, 0], self.XY[cond==True, 1], 'sb', label='Pinned in x2')
        #Free
        cond = 1 - (self.pins[:, 0] + self.pins[:, 1])
        if sum(cond):
            plt.plot(self.XY[cond==True, 0], self.XY[cond==True, 1], 'sk', label='Free')

        for i in range(len(self.ELS)):
            plt.fill(self.XY[self.ELS[i, :], 0], self.XY[self.ELS[i, :], 1], edgecolor='k', fill=False)


        if show_ids:
            for i in range(4):                             #loop over all nodes within an element
                for j in range(len(self.ELS)):                  #loop over all elements
                    sh=0.01
                    try: 
                        plt.text(self.XY[self.ELS[j,i],0]+sh,self.XY[self.ELS[j,i],1]+sh, self.ELS[j,i])
                    except:
                        pass

        # Set chart title.
        plt.title(title, fontsize=19)
        # Set x axis label.
        plt.xlabel("$x_1$", fontsize=10)
        # Set y axis label.
        plt.ylabel("$x_2$", fontsize=10)

        plt.legend()

        plt.show()

    def copy(self):
        mesh = Mesh(self.corners, self.nx, self.ny, self.node_dist)

        mesh.all_corners = self.all_corners
        mesh.XY = self.XY
        mesh.ELS = self.ELS
        mesh.DOF = self.DOF
        mesh.pins = self.pins
        mesh.forces = self.forces
        return mesh
    
    def __add__(self, mesh2):
        mesh1 = self.copy()
        mesh1.pins = self.pins
        mesh1.forces = self.forces
        mesh1 += mesh2
        return mesh1
    
    def __iadd__(self, mesh2):
        '''
        Glue two meshes together via overlapping nodes

        Requires there to be a shared edge, with same number of nodes in that direction
        Ie, two shared corners, and one of nx or ny to be the same

        Also requires the coord_func to be the same
        '''
        # Check if coord_func bias method is the same
        assert self.node_dist == mesh2.node_dist
        # Find shared edge
        mesh1_shared, mesh2_shared = shared_pairs(self.XY, mesh2.XY)

        mesh2_ELS_copy = mesh2.ELS.copy()

        offset = np.max(self.ELS) + 1
        for i in range(len(mesh2_ELS_copy)):
            for j in range(4):
                # Replace node ids in mesh2 with duplicates in self
                mask = (mesh2_shared == mesh2_ELS_copy[i, j])
                if np.sum(mask):
                    # ELS[i, j] is a shared node
                    mesh2_ELS_copy[i, j] = mesh1_shared[mask]
                else:
                    # no shared nodes
                    mesh2_ELS_copy[i, j] += offset
        
        
        for i, node in enumerate(mesh2_shared[::-1]):
            # Account for missing IDS (shift)
            mesh2_ELS_copy[mesh2_ELS_copy > node + offset] -= 1


        mask = np.ones(mesh2.XY.shape[0], dtype=bool)
        mask[mesh2_shared] = False

        total_XY = np.append(self.XY, mesh2.XY[mask], axis=0) # mask out duplicated nodes

        # Merge pins
        self.pins = np.append(self.pins, mesh2.pins[mask], axis=0)

        #Merge forces
        self.forces = np.append(self.forces, mesh2.forces[mask], axis=0)
        
        #mask = np.ones(mesh2.XY.shape[0], dtype=bool)
        #total_XY = np.append(self.XY, mesh2.XY, axis=0)
        total_ELS = np.append(self.ELS, mesh2_ELS_copy, axis=0)
        self.XY = total_XY
        self.ELS = total_ELS

        nnodes = len(total_ELS[:, 0])
        self.DOF = np.zeros((nnodes, 8), dtype=int)

        # Regenerate dofs
        for i in range(nnodes):
            self.DOF[i, :] = [self.ELS[i,0]*2, self.ELS[i,1]*2-1, self.ELS[i,1]*2, self.ELS[i,1]*2+1, self.ELS[i,2]*2, self.ELS[i,2]*2+1, self.ELS[i,3]*2, self.ELS[i,3]*2+1]

        self.all_corners = np.unique(np.append(self.all_corners, mesh2.all_corners, axis=0), axis=0)

        return self

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




