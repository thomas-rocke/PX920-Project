import numpy as np
import matplotlib.pyplot as plt
from MicroMesh import MicroMesh, MicroSolver


def Converge(mat_1_params=[10E9, 0.32], mat_2_params=[80E9, 0.22], rel_conc=0.55, nxs=np.array([3, 4, 5, 7, 10, 15, 20, 28, 44, 50])):
    E_1, nu_1 = mat_1_params
    E_2, nu_2 = mat_2_params

    Cs = np.zeros((nxs.shape[0], 3, 3))
    real_concs = np.zeros((nxs.shape[0]))
    nnodes = np.zeros_like(real_concs)

    for i, nx in enumerate(nxs):
        mesh = MicroMesh(nx, nx, E_1, E_2, nu_1, nu_2, rel_conc)
        solver = MicroSolver(mesh)
        Cs[i, :, :] = solver.homogenize()
        real_concs[i] = np.sum([el.E == E_1 for el in mesh.ELS])/mesh.ELS.shape[0]
        nnodes[i] = mesh.nnodes

    return Cs, real_concs, nnodes


def plot_c_converge(nnodes, Cs, mat_1_params=[10E9, 0.32], mat_2_params=[80E9, 0.22], rel_conc=0.55):
    E_1, nu_1 = mat_1_params
    E_2, nu_2 = mat_2_params
    mesh = MicroMesh(3, 3, E_1, E_2, nu_1, nu_2, rel_conc)
    solver = MicroSolver(mesh)

    voigt_C = solver.Voigt()/1E9
    reuss_C = solver.Reuss()/1E9
    C_1 = solver.elasticity(E_1, nu_1)/1E9
    C_2 = solver.elasticity(E_2, nu_2)/1E9

    Cs = Cs.copy()/1E9

    fig, ax = plt.subplots(3, 3)

    for i in range(3):
        for j in range(3):
            ax[i, j].plot(nnodes, Cs[:, i, j], label="Homogenized", color='k')
            ax[i, j].hlines(voigt_C[i, j], 0, np.max(nnodes), label='Voigt bound', color='C1', linestyle='dashed')
            ax[i, j].hlines(reuss_C[i, j], 0, np.max(nnodes), label='Reuss bound', color='C2', linestyle='dashed')
            #ax[i, j].hlines(C_1[i, j], 0, np.max(nnodes), label='Material 1', color='C3', linestyle='dashed')
            #ax[i, j].hlines(C_2[i, j], 0, np.max(nnodes), label='Material 2', color='C4', linestyle='dashed')
            ax[i, j].set_title(f"Convergence of $C_{{ {i + 1}, {j + 1} }}$")
            ax[i, j].set_xlabel("Total Number of Nodes")
            ax[i, j].set_ylabel("Estimated Elastic Property/GPa")
            ax[i, j].set_xscale('log')
            ax[i, j].legend()

    plt.show()




Cs, real_concs, nnodes = Converge()
plot_c_converge(nnodes, Cs)


    