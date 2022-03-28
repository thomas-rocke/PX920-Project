'''
Set of functions to generate real-space nodal coordinates given bounds and number of nodes

'''

import numpy as np

def shapes(xi, eta):
    '''Nodal functions; returns a vector as a function of xi, eta'''
    N1 = 0.25*(1.0-xi)*(1.0+eta)
    N2 = 0.25*(1.0-xi)*(1.0-eta)
    N3 = 0.25*(1.0+xi)*(1.0-eta)
    N4 = 0.25*(1.0+xi)*(1.0+eta)
    return np.array([N1, N2, N3, N4])


def Transform(corners, Meshgrid, nx, ny):
    '''
    Transforms the square Meshgrid to fit the quadrilateral defined by corners
    Then converts to a form usable by the Mesh class
    '''
    # Unpack
    x, y = Meshgrid

    coords = shapes(x, y).T @ corners
    x = coords[:, :, 0]
    y = coords[:, :, 1]
    
    XY = np.zeros((nx * ny, 2))

    for i in range(nx):
        for j in range(ny):
            XY[i + nx * j, :] = [x[i, j], y[i, j]]
    return XY




def Uniform(corners, nx, ny):
    '''
    Generates a uniformly distributed meshgrid for the quadrilateral defined by corners
    '''
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    grid = np.meshgrid(x, y)
    return Transform(corners, grid, nx, ny)


def CornerBias(corners, nx, ny):
    '''
    Generates a meshgrid biased towards the corners of the domain
    '''
    # Generate unit grid
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    l = 0.5

    # Apply sqrt biasing, preserving sign
    x = x * np.sqrt(np.abs(x)/l) / np.abs(x + 1E-20)
    y = y * np.sqrt(np.abs(y)/l) / np.abs(y + 1E-20)

    x /= np.max(x) - np.min(x)
    y /= np.max(y) - np.min(y)


    # Enforce -1,1 meshgrid
    x += np.min(x)
    x *= 2
    x -= 1

    y += np.min(y)
    y *= 2
    y -= 1

    grid = np.meshgrid(x, y)
    return Transform(corners, grid, nx, ny)


