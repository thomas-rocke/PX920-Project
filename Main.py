from FEMSolver import FEM
from NodeDists import *
from Meshing import Mesh
import numpy as np
import matplotlib.pyplot as plt
from SampleShapes import *
from PlottingUtils import *
from scipy.signal import correlate

def converge(shape, ns, mpoints):

    disp_errs = np.zeros((mpoints, mpoints, 2, len(ns)))
    force_errs = np.zeros_like(disp_errs)
    
    stress_errs = np.zeros((mpoints, mpoints, 3, len(ns)))
    strain_errs = np.zeros_like(stress_errs)

    nnodes = np.zeros((len(ns)))

    for i, n in enumerate(ns):
        print(f'Solving n={n} shape')
        solver = shape(n)
        solver.solve()
        #solver.show_deformation(magnification=10000)
        solver.get_props(mpoints=mpoints)

        stress_errs[:, :, :, i] = solver.Stresses
        strain_errs[:, :, :, i] = solver.Strains
        disp_errs[:, :, :, i] = solver.Displacements
        force_errs[:, :, :, i] = solver.Forces

        nnodes[i] = solver.mesh.XY.shape[0]


    stress_errs -= stress_errs[:, :, :, -1][:, :, :, None]
    strain_errs -= strain_errs[:, :, :, -1][:, :, :, None]
    disp_errs -= disp_errs[:, :, :, -1][:, :, :, None]
    force_errs -= force_errs[:, :, :, -1][:, :, :, None]
    #errs /= errs[:, :, :, -1][:, :, :, None] + 1.0E-40

    #x, y = solver.coords

    #labels = ["$\epsilon_{11}$ error", "$\epsilon_{22}$ error", "$\gamma_{12}$ error", "$|\epsilon|$ error"]

    #corners = solver.mesh.all_corners

    #corners = np.append(corners, corners[0, :][None, :], axis=0)

    #plotting_3(x, y, errs[:, :, :, 0], labels, corners)

    abs_stress_err = np.linalg.norm(stress_errs, axis=-2)
    abs_strain_err = np.linalg.norm(strain_errs, axis=-2)
    abs_disp_err = np.linalg.norm(disp_errs, axis=-2)
    abs_force_err = np.linalg.norm(force_errs, axis=-2)
    
    
    stress_err = np.average(abs_stress_err.reshape(-1, abs_stress_err.shape[-1]), axis=0)
    strain_err = np.average(abs_strain_err.reshape(-1, abs_strain_err.shape[-1]), axis=0)
    disp_err = np.average(abs_disp_err.reshape(-1, abs_disp_err.shape[-1]), axis=0)
    force_err = np.average(abs_force_err.reshape(-1, abs_force_err.shape[-1]), axis=0)

    plt.plot(nnodes[:-1], stress_err[:-1])
    plt.yscale("log")
    plt.xlabel("Number of Nodes")
    plt.ylabel(f"Error on normed Stresses")
    plt.title(f"Error convergence of Stress with increasing mesh complexity")
    plt.show()

    plt.plot(nnodes[:-1], strain_err[:-1])
    plt.yscale("log")
    plt.xlabel("Number of Nodes")
    plt.ylabel(f"Error on normed Strains")
    plt.title(f"Error convergence of Strain with increasing mesh complexity")
    plt.show()

    plt.plot(nnodes[:-1], disp_err[:-1])
    plt.yscale("log")
    plt.xlabel("Number of Nodes")
    plt.ylabel(f"Error on normed Displacements")
    plt.title(f"Error convergence of Displacement with increasing mesh complexity")
    plt.show()

    plt.plot(nnodes[:-1], force_err[:-1])
    plt.yscale("log")
    plt.xlabel("Number of Nodes")
    plt.ylabel(f"Error on normed Forces")
    plt.title(f"Error convergence of Force with increasing mesh complexity")
    plt.show()

    return nnodes[:-1], stress_err[:-1]



mpoints = 2500

ns_c = np.unique(np.logspace(np.log10(2), np.log10(30), 10).astype(int))
ns_i = np.unique(np.logspace(np.log10(2), np.log10(20), 10).astype(int))

'''
c_nodes, c_err = converge(C_shape, ns_c, mpoints)
i_nodes, i_err = converge(I_shape, ns_i, mpoints)

ref_n = np.logspace(1, np.log10(500), 1000)
ref_n_2 = np.logspace(1, 3, 1000)
ref_e = 10/ref_n**2
ref_e_2 = 3/ref_n_2

plt.plot(c_nodes, c_err, label='Symmetry Reduced Shape')
plt.plot(i_nodes, i_err, label='Full Shape')
plt.plot(ref_n, ref_e, label="$\mathcal{O}(n^{-2})$ reference line", linestyle='dashed')
plt.plot(ref_n, ref_e_2, label="$\mathcal{O}(n^{-1})$ reference line", linestyle='dashed')
plt.title("Comparison of Stress Convergence")
plt.xlabel("Number of Nodes")
plt.ylabel("Error on Normed Stresses")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()

'''
#solver = simple_shape(100)
#print(solver.mesh.XY.shape[0])
#solver.mesh.plot()
#solver.solve()
#solver.get_props()
#solver.plot_props()

nnodes = np.zeros_like(ns_c)
max_disp = np.zeros_like(ns_c, dtype=float)

for i, n in enumerate(ns_c):
    solver = C_shape(n)
    nnodes[i] = solver.mesh.XY.shape[0]
    solver.solve()
    solver.get_props(mpoints=mpoints)
    max_disp[i] = np.max(np.linalg.norm(solver.Displacements, axis=-1))


elem_size = (np.max(solver.mesh.XY[:, 1]) - np.min(solver.mesh.XY[:, 1]))/nnodes

plt.plot(elem_size, max_disp)
plt.title("Variation of Maximum Displacement with Element Size")
plt.xlabel("Average Vertical Side Length of Elements")
plt.ylabel("Maximum |d|")
plt.xscale("log")
plt.yscale("log")
plt.show()
'''
coords = np.array([[0, 0],
                   [0, 0.7],
                   [0.25, 0.7],
                   [0.25, 0.6],
                   [0.1, 0.6], 
                   [0.1, 0.1],
                   [0.25, 0.1],
                   [0.25, 0]])

coords = np.array([[0.25, 0],
                   [0.25, 0.1],
                   [0.1, 0.1],
                   [0.1, 0.6],
                   [0.25, 0.6],
                   [0.25, 0.7],
                   [-0.25, 0.7],
                   [-0.25, 0.6],
                   [-0.1, 0.6],
                   [-0.1, 0.1],
                   [-0.25, 0.1],
                   [-0.25, 0],
                   [0.25, 0]])

solver = I_shape(15)
#print(solver.mesh.ELS)
#solver.mesh.plot()
solver.solve()
#solver.show_deformation(10**11)
solver.get_props(lin_samples_per_elem=8, mpoints=2501)

#plt.plot(coords[:, 0], coords[:, 1])
#plt.show()
solver.plot_props(shape_outline=coords)'''
