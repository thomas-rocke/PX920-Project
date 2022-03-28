from Meshing import Mesh
from FEMSolver import FEM
from NodeDists import Uniform
import numpy as np


def simple_shape(n):
    '''
    Simple Quadrilateral test case
    '''
    corners = np.array([[0, 1], # top left
                    [2, 1], # top right
                    [2, 0.5], # bottom right
                    [0, 0]]) # bottom left

    mesh = Mesh(corners, n, n, coord_func=Uniform)

    force = np.array([0, 20])
    edge='top'

    solver = FEM(mesh, 3E7, 0.3, quad_points=5)
    mesh.apply_load(force, edge)
    mesh.pin_edge('left', 0)
    mesh.pin_edge('left', 1)

    return solver


def C_shape(n):
    '''
    C shaped solid
    
    '''
    func = Uniform
    E = 80.0E9
    nu = 0.3



    corners1 = np.array([[0, 0.1],
                        [0.1, 0.1],
                        [0.1, 0],
                        [0, 0]])
    mesh1 = Mesh(corners1, n, n, func)
    mesh1.pin_edge('bottom', 0)
    mesh1.pin_edge('bottom', 1)
    mesh1.pin_edge('left', 0)

    corners2 = np.array([[0.1, 0.1], 
                        [0.25, 0.1],
                        [0.25, 0],
                        [0.1, 0]])
    mesh2 = Mesh(corners2, n, n, func)
    mesh2.pin_edge('bottom', 0)
    mesh2.pin_edge('bottom', 1)

    corners3 = np.array([[0, 0.2],
                        [0.1, 0.2],
                        [0.1, 0.1],
                        [0, 0.1]])
    mesh3 = Mesh(corners3, n, n, func)
    mesh3.pin_edge('left', 0)

    corners3a = np.array([[0, 0.5],
                        [0.1, 0.5],
                        [0.1, 0.2],
                        [0, 0.2]])
    mesh3a = Mesh(corners3a, n, n, func)
    mesh3a.pin_edge('left', 0)

    corners3b = np.array([[0, 0.6],
                        [0.1, 0.6],
                        [0.1, 0.5],
                        [0, 0.5]])
    mesh3b = Mesh(corners3b, n, n, func)
    mesh3b.pin_edge('left', 0)

    corners4 = np.array([[0, 0.7], 
                        [0.1, 0.7],
                        [0.1, 0.6],
                        [0, 0.6]])
    mesh4 = Mesh(corners4, n, n, func)
    mesh4.pin_edge('left', 0)
    mesh4.apply_load([0, 100], 'top')

    corners5 = np.array([[0.1, 0.7],
                        [0.25, 0.7],
                        [0.25, 0.6],
                        [0.1, 0.6]])
    mesh5 = Mesh(corners5, n, n, func)
    mesh5.apply_load([0, 100], 'top')

    shape_outline = np.array([[0, 0],
                            [0, 0.7],
                            [0.25, 0.7],
                            [0.25, 0.6],
                            [0.1, 0.6],
                            [0.1, 0.1],
                            [0.25, 0.1],
                            [0.25, 0],
                            [0, 0]])



    #mesh4.plot()

    #print(np.sum(mesh4.pins))

    s4 = mesh4 + mesh5
    s4a = mesh3b + s4
    s4b = mesh3a + s4a
    #print(np.sum(s4.pins))
    s3 = mesh3 + s4b
    s2 = mesh1 + s3
    shape = mesh2 + s2
    #shape.plot(show_ids=False)

    solver = FEM(shape, E, nu)
    return solver

def I_shape(n):
    '''
    I shaped solid
    '''
    func = Uniform
    E = 80.0E9
    nu = 0.3

    mesh1 = C_shape(n).mesh

    mesh2 = C_shape(n).mesh

    mesh2.XY[:, 0] *= -1 # Flip the x coords

    mesh = mesh1 + mesh2 # Glue flipped c to c

    y_min = np.min(mesh.XY[:, 1])

    non_bottom = (mesh.XY[:, 1] != y_min) # Find all nodes that are not on the lower edge

    
    mesh.pins[non_bottom] = False # Unpin everything but the bottom nodes

    solver = FEM(mesh, E, nu)
    return solver