import numpy as np
import matplotlib.pyplot as plt
from MicroMesh import MicroMesh, MicroSolver
from tqdm import tqdm


def int_logscale(max_n):
    init_scale = np.logspace(np.log10(3), np.log10(max_n), num=30)
    return np.unique(init_scale.astype(int))

def Converge(mat_1_params=[10E9, 0.32], mat_2_params=[80E9, 0.22], rel_conc=0.55, nxs=np.arange(3, 8), num_candidates=10):
    E_1, nu_1 = mat_1_params
    E_2, nu_2 = mat_2_params

    Cs = np.zeros((nxs.shape[0], 3, 3))
    C_errs = np.zeros_like(Cs)
    temp_Cs = np.zeros((num_candidates, 3, 3))
    real_concs = np.zeros((nxs.shape[0]))
    nnodes = np.zeros_like(real_concs)

    Es = np.zeros_like(nnodes)
    E_errs = np.zeros_like(nnodes)
    E_tmp = np.zeros((num_candidates))

    nus = np.zeros_like((Es))
    nu_errs = np.zeros_like((nus))
    nu_tmp = np.zeros_like(E_tmp)
    print(f"Using nx values of {nxs}")
    for i in tqdm(range(len(nxs)), desc="Running through nx values"):
        nx = nxs[i]
        for j in tqdm(range(num_candidates), desc="Testing ensemble of Meshes", leave=False):
            mesh = MicroMesh(nx, nx, E_1, E_2, nu_1, nu_2, rel_conc)
            solver = MicroSolver(mesh)
            temp_Cs[j, :, :] = solver.homogenize()
            E_tmp[j], nu_tmp[j] = solver.infer_props()
        Cs[i, :, :] = np.average(temp_Cs, axis=0)
        C_errs[i, :, :] = np.average((temp_Cs - Cs[i, :, :])**2)

        Es[i] = np.average(E_tmp)
        E_errs[i] = np.average((E_tmp - Es[i])**2)

        nus[i] = np.average(nu_tmp)
        nu_errs[i] = np.average((nu_tmp - nus[i])**2)

        real_concs[i] = np.sum([el.E == E_1 for el in mesh.ELS])/mesh.ELS.shape[0]
        nnodes[i] = mesh.nnodes


    C_errs = np.sqrt(C_errs)
    E_errs = np.sqrt(E_errs)
    nu_errs = np.sqrt(nu_errs)

    return Cs, C_errs, Es, E_errs, nus, nu_errs, real_concs, nnodes



def plot_c_converge(Cs, C_errs, Es, E_errs, nus, nu_errs, real_concs, nnodes, mat_1_params=[10E9, 0.32], mat_2_params=[80E9, 0.22], rel_conc=0.55):
    E_1, nu_1 = mat_1_params
    E_2, nu_2 = mat_2_params
    mesh = MicroMesh(3, 3, E_1, E_2, nu_1, nu_2, rel_conc)
    solver = MicroSolver(mesh)

    voigt_C = solver.Voigt()/1E9
    reuss_C = solver.Reuss()/1E9
    C_1 = solver.elasticity(E_1, nu_1)/1E9
    C_2 = solver.elasticity(E_2, nu_2)/1E9

    Cs = Cs.copy()/1E9
    C_errs = C_errs.copy()/1E9

    Es = Es.copy()/1E9
    E_errs = E_errs/1E9

    fig, ax = plt.subplots(3, 3, figsize=(14.0, 14.0))
    
    
    font_size=20

    for i in range(3):
        for j in range(3):
            ax[i, j].plot(nnodes, Cs[:, i, j], label="Homogenized", color='k')
            ax[i, j].fill_between(nnodes, Cs[:, i, j] + 2*C_errs[:, i, j], Cs[:, i, j] - 2*C_errs[:, i, j], color='C3', alpha=0.3)
            ax[i, j].hlines(voigt_C[i, j], 0, np.max(nnodes), label='Voigt bound', color='C1', linestyle='dashed')
            ax[i, j].hlines(reuss_C[i, j], 0, np.max(nnodes), label='Reuss bound', color='C2', linestyle='dashed')
            #ax[i, j].hlines(C_1[i, j], 0, np.max(nnodes), label='Material 1', color='C3', linestyle='dashed')
            #ax[i, j].hlines(C_2[i, j], 0, np.max(nnodes), label='Material 2', color='C4', linestyle='dashed')
            ax[i, j].set_title(f"Convergence of $C_{{{i + 1},{j + 1}}}$")
            ax[i, j].set_xlabel("Total Number of Nodes")
            ax[i, j].set_ylabel("Estimated Elastic Property/GPa")
            ax[i, j].set_xscale('log')

            # if (i<2 and j<2) or (i==3 and j==3):
            #     ax[i, j].set_yscale("log")
            ax[i, j].legend()
    plt.tight_layout()      
    plt.savefig("C_convergence.png")

    E_voigt, nu_voigt = solver.infer_props(voigt_C*1E9)
    E_reuss, nu_reuss = solver.infer_props(reuss_C*1E9)

    E_voigt /= 1E9
    E_reuss /= 1E9

    fig, ax = plt.subplots(nrows=3, figsize=(14.0, 7.0))

    ax[0].plot(nnodes, Es, label="Homogenized", color='k')
    ax[0].fill_between(nnodes, Es + 2*E_errs, Es - 2*E_errs, color='C3', alpha=0.3)
    ax[0].hlines(E_voigt, 0, np.max(nnodes), label='Voigt bound', color='C1', linestyle='dashed')
    ax[0].hlines(E_reuss, 0, np.max(nnodes), label='Reuss bound', color='C2', linestyle='dashed')
    ax[0].set_title("Convergence of $E_{eff}$")
    ax[0].set_xlabel("Total Number of Nodes")
    ax[0].set_ylabel("Estimated\nYoung's Modulus/GPa")
    ax[0].set_xscale('log')
    ax[0].legend()


    ax[1].plot(nnodes, nus, label="Homogenized", color='k')
    ax[1].fill_between(nnodes, nus + 2*nu_errs, nus - 2*nu_errs, color='C3', alpha=0.3)
    ax[1].hlines(nu_voigt, 0, np.max(nnodes), label='Voigt bound', color='C1', linestyle='dashed')
    ax[1].hlines(nu_reuss, 0, np.max(nnodes), label='Reuss bound', color='C2', linestyle='dashed')
    ax[1].set_title(r"Convergence of $\nu_{eff}$")
    ax[1].set_xlabel("Total Number of Nodes")
    ax[1].set_ylabel("Estimated\nPoisson's Ratio")
    ax[1].set_xscale('log')
    ax[1].legend()

    ax[2].plot(nnodes, real_concs, label="Simulated", color='k')
    ax[2].hlines(rel_conc, 0, np.max(nnodes), label='Target', color='C1', linestyle='dashed')
    ax[2].set_title("Convergence of Relative Concentration")
    ax[2].set_xlabel("Total Number of Nodes")
    ax[2].set_ylabel("Relative Concentration\nof Materials")
    ax[2].set_xscale('log')
    ax[2].legend()

    plt.tight_layout()      
    plt.savefig("E_convergence.png")

def sens_analysis(steps=np.array([10E7, 80E7, 0.0032, 0.0022, 0.006]), x_opt=np.array([10E9, 80E9, 0.32, 0.22, 0.55])):
    def central_difference(Q_1, Q_2, step):
        return (Q_2 - Q_1)/(2*step)

    Q_grads = np.zeros((steps.shape[0], 3, 3))

    for i in tqdm(range(steps.shape[0]), desc="Generating Sens for parameters"):
        x = x_opt
        x[i] -= steps[i]

        mesh = mesh(*x)
        solver = solver(mesh)
        Q_1 = solver.homogenize()

        x = x_opt
        x[i] += steps[i]

        mesh = mesh(*x)
        solver = solver(mesh)
        Q_2 = solver.homogenize()

        Q_grads[i, :, :] = central_difference(Q_1, Q_2, steps[i]) * x_opt # Scaled Sensitivities
    return Q_grads


def plot_sens(Q_grads):
    fig, ax = plt.subplots(3, 3)

    labels=[r"E$_1$", r"E$_2$", r"$\nu_1$", r"$\nu_2$", r"Volume Fraction"]

    for i in range(3):
        for j in range(3):
            ax[i, j].bar(range(Q_grads.shape[0]), Q_grads[:, i, j])
            ax[i, j].set_xticks(range(Q_grads.shape[0], labels))
            ax[i, j].set_xlabel("Parameter")
            ax[i, j].set_ylabel("Scaled Sensitivity")
            ax[i, j].set_title(f"Sensitivity of $C_{{{i + 1},{j + 1}}}$")
    plt.tight_layout()

    plt.savefig("Sensitivities.png")




# props = Converge()
# plot_c_converge(*props)

    