%%time
def wigner_laguerre(rho, x_min, x_max, p_min, p_max, x_res = 200, p_res = 200):
    """A very basic code for obtaining the wigner distribution from the density matrix. 
    Refs:
    1. Ulf leonhardt - Measuring the Quantum States of Light. Chapter 5, Section 5.2.6 pg.nos 128-129
    2. "Numerical study of Wigner negativity in one-dimensional steady-state resonance fluorescence"
    arXiv: 1909.02395v1 Appendix A
    """
    if type(rho) != np.array:
        rho = np.array(rho)
    if rho.shape[0] == rho.shape[1]:
        x_vec = np.linspace(x_min, x_max, x_res)
        p_vec = np.linspace(p_min, p_max, p_res)
        X, P = np.meshgrid(x_vec, p_vec)
        A = X + 1j * P
        B = np.abs(A)**2
        W = np.zeros((x_res, p_res))
        for n in range(rho.shape[0]):
            if np.abs(rho[n, n]) > 0:
                W += np.real(rho[n, n] * (-1) ** n * genlaguerre(n, 0)(2 * B))
            for m in range(0, n):
                if np.abs(rho[m, n]) > 0:
                    W += 2 * np.real(rho[m, n] * (-1) ** m * np.sqrt(2 ** (n - m) * factorial(m) / factorial(n)) * genlaguerre(m, n - m)(2 * B) * A ** (n - m))
        W = W * np.exp(-B)  / np.pi
        return W / np.sum(W)
    else:
        raise ValueError("Dim. mismatch between rows and columns")
rho = sq.density_mat_from_state_vec([1 for x in range(3)])         
ml = wigner_laguerre(rho, -15, 15, -15, 15, 500, 500)
np.min(ml)
