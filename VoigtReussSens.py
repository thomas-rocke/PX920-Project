import sympy as sy
import numpy as np

def Plane_Strain(E, nu):
    '''C matrix in the case of plane strain'''
    const = E/((1.0-2*nu)/(1 + nu))
    C = sy.Matrix([[1.0 - nu, nu, 0.0], [nu, 1.0-nu, 0.0], [0.0, 0.0, 0.5*(1.0-2 * nu)]])
    return const*C

def Voigt(C_1, C_2, frac):

    return frac * C_1 + (1-frac) * C_2

def Reuss(C_1, C_2, frac):
    res = sy.Matrix([[1/(frac / C_1[i, j] + (1-frac) / C_2[i, j]) for i in range(3)] for j in range(3)])
    res[2, 0] = 0; res[2, 1] = 0
    res[0, 2] = 0; res[1, 2] = 0
    return res


def get_VR_sens():
    E1, nu1, E2, nu2, vol_frac = sy.symbols("E1 nu1 E2 nu2 phi")

    ps1 = Plane_Strain(E1, nu1)

    ps2 = Plane_Strain(E2, nu2)


    v = Voigt(ps1, ps2, vol_frac)
    r = Reuss(ps1, ps2, vol_frac)


    mat_1_params=[10E9, 0.32]
    mat_2_params=[80E9, 0.22]
    rel_conc = 0.55

    E_1, nu_1 = mat_1_params
    E_2, nu_2 = mat_2_params

    vars = [E1, E2, nu1, nu2, vol_frac]
    var_vals = [E_1, E_2, nu_1, nu_2, rel_conc]
    v_diffs = np.array([np.array(v.diff(var).subs(E1, E_1).subs(E2, E_2).subs(nu1, nu_1).subs(nu2, nu_2).subs(vol_frac, rel_conc)) for var in [E1, E2, nu1, nu2, vol_frac]])
    r_diffs = np.array([np.array(r.diff(var).subs(E1, E_1).subs(E2, E_2).subs(nu1, nu_1).subs(nu2, nu_2).subs(vol_frac, rel_conc)) for var in [E1, E2, nu1, nu2, vol_frac]])

    for i in range(5):
        v_diffs[i] *= var_vals[i]
        r_diffs[i] *= var_vals[i]
    return v_diffs, r_diffs