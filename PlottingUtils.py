import numpy as np
import matplotlib.pyplot as plt



def mask_outside_polygon(poly_verts, ax=None):
    """
    Plots a mask on the specified axis ("ax", defaults to plt.gca()) such that
    all areas outside of the polygon specified by "poly_verts" are masked.  

    "poly_verts" must be a list of tuples of the verticies in the polygon in
    counter-clockwise order.

    Returns the matplotlib.patches.PathPatch instance plotted on the figure.
    """
    import matplotlib.patches as mpatches
    import matplotlib.path as mpath

    if ax is None:
        ax = plt.gca()

    # Get current plot limits
    xlim = np.array(ax.get_xlim())*1.1
    ylim = np.array(ax.get_ylim())*1.1

    # Verticies of the plot boundaries in clockwise order
    outside_vertices = np.array([[xlim[0], ylim[0]],
                                    [xlim[0], ylim[1]],
                                    [xlim[1], ylim[1]],
                                    [xlim[1], ylim[0]],
                                    [xlim[0], ylim[0]]])[:, :, None]

    outside_vertices = np.hstack((outside_vertices[:, 0, :], outside_vertices[:, 1, :]))


    inside_vertices = np.hstack((poly_verts[:, 0][:, None], poly_verts[:, 1][:, None]))

    ins_codes = np.ones(
        len(inside_vertices), dtype=mpath.Path.code_type) * mpath.Path.LINETO
    ins_codes[0] = mpath.Path.MOVETO

    ots_codes = np.ones(
        len(outside_vertices), dtype=mpath.Path.code_type) * mpath.Path.LINETO
    ots_codes[0] = mpath.Path.MOVETO

    # Concatenate the inside and outside subpaths together, changing their
    # order as needed
    vertices = np.concatenate((outside_vertices[::1],
                            inside_vertices[::-1]))
    # Shift the path
    #vertices[:, 0] += i * 2.5
    # The codes will be all "LINETO" commands, except for "MOVETO"s at the
    # beginning of each subpath
    all_codes = np.concatenate((ots_codes, ins_codes))
    # Create the Path object
    path = mpath.Path(vertices, all_codes)
    # Add plot it
    patch = mpatches.PathPatch(path, facecolor='blue', edgecolor='black')
    ax.add_patch(patch)

    return patch

def plotting_3(x, y, data, labels, corners):
    fig, ax = plt.subplots(nrows=2, ncols = 2)

    #poly2 = mask_outside_polygon(corners, ax[1, 0])
    #poly3 = mask_outside_polygon(corners, ax[0, 1])
    #poly4 = mask_outside_polygon(corners, ax[1, 1])
    vmax = np.max(np.abs(data))
    vmin = -vmax

    xmin = np.min(x)
    xmax = np.max(x)

    ymin = np.min(y)
    ymax = np.max(y)

    col = ax[0, 0].imshow(data[:, :, 0], extent=(xmin, xmax, ymin, ymax), vmin=vmin, vmax=vmax, cmap='bwr', origin='lower')
    ax[0, 0].set_title(labels[0])
    mask_outside_polygon(corners, ax[0, 0])


    ax[1, 0].imshow(data[:, :, 1], extent=(xmin, xmax, ymin, ymax), vmin=vmin, vmax=vmax, cmap='bwr', origin='lower')
    ax[1, 0].set_title(labels[1])
    mask_outside_polygon(corners, ax[1, 0])

    ax[0, 1].imshow(data[:, :, 2], extent=(xmin, xmax, ymin, ymax), vmin=vmin, vmax=vmax, cmap='bwr', origin='lower')
    ax[0, 1].set_title(labels[2])
    mask_outside_polygon(corners, ax[0, 1])

    ax[1, 1].imshow(np.linalg.norm(data, axis=-1), extent=(xmin, xmax, ymin, ymax), vmin=vmin, vmax=vmax, cmap='bwr', origin='lower')
    ax[1, 1].set_title(labels[3])
    mask_outside_polygon(corners, ax[1, 1])
    
    
    fig.subplots_adjust(right=0.4)
    cbar_ax = fig.add_axes([0.45, 0.15, 0.05, 0.7])
    fig.colorbar(col, cax=cbar_ax)
    plt.show()

def plotting_2(x, y, data, labels, corners):
    fig, ax = plt.subplots(nrows=2, ncols = 2)

    #poly2 = mask_outside_polygon(corners, ax[1, 0])
    #poly3 = mask_outside_polygon(corners, ax[0, 1])
    #poly4 = mask_outside_polygon(corners, ax[1, 1])
    vmax = np.max(np.abs(data))
    vmin = -vmax

    xmin = np.min(x)
    xmax = np.max(x)

    ymin = np.min(y)
    ymax = np.max(y)

    col = ax[0, 0].imshow(data[:, :, 0], extent=(xmin, xmax, ymin, ymax), vmin=vmin, vmax=vmax, cmap='bwr', origin='lower')
    ax[0, 0].set_title(labels[0])
    mask_outside_polygon(corners, ax[0, 0])


    ax[1, 0].imshow(data[:, :, 1], extent=(xmin, xmax, ymin, ymax), vmin=vmin, vmax=vmax, cmap='bwr', origin='lower')
    ax[1, 0].set_title(labels[1])
    mask_outside_polygon(corners, ax[1, 0])

    ax[0, 1].imshow(np.linalg.norm(data, axis=-1), extent=(xmin, xmax, ymin, ymax), vmin=vmin, vmax=vmax, cmap='bwr', origin='lower')
    ax[0, 1].set_title(labels[2])
    mask_outside_polygon(corners, ax[0, 1])
    
    ax[1, 1].axis('off')
    
    fig.subplots_adjust(right=0.4)
    cbar_ax = fig.add_axes([0.45, 0.15, 0.05, 0.7])
    fig.colorbar(col, cax=cbar_ax)
    plt.show()