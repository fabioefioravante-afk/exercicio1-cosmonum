from pathlib import Path
import numpy as np
from scipy.linalg import solve_triangular

PASTA_DADOS = Path(__file__).parent / "data"
N_SN = 580

def carrega_catalogo():
    return np.loadtxt(PASTA_DADOS / "SCPUnion2.1_mu_vs_z.txt", usecols=(1, 2, 3), unpack=True)

def carrega_covariancia(sistematicas=True):
    nome = "SCPUnion2.1_covmat_sys.txt" if sistematicas else "SCPUnion2.1_covmat_nosys.txt"
    return np.loadtxt(PASTA_DADOS / nome)

z_sn, mu_obs, sigma_mu = carrega_catalogo()
cov_sys = carrega_covariancia()
L_sys = np.linalg.cholesky(cov_sys)

uns = np.ones(N_SN)
w = solve_triangular(L_sys, uns, lower=True) # L_sys @ w = 1
C1 = w @ w

def chi2_marg(mu_teoria):
    #README, seção 4.4
    delta = mu_obs - mu_teoria
    y = solve_triangular(L_sys, delta, lower=True) #resolve L_sys @ y = delta
    A = y @ y
    B = y @ w
    return A - B**2 / C1

def log_likelihood(theta, model):
    omega_m, omega_lambda = theta
    with np.errstate(invalid="ignore", divide="ignore"):
        cosmo = model(H0=70, omega_m=omega_m, omega_lambda=omega_lambda)
        mu_teoria = cosmo.mu(z_sn)
      # distancia invalida (a luz passou do polo oposto de um universo fechado)
    if not np.all(np.isfinite(mu_teoria)):
        return -np.inf
    return -0.5 * chi2_marg(mu_teoria)

OM_MIN, OM_MAX = 0.0, 2.0
OL_MIN, OL_MAX = -1.0, 3.0
z_checagem = np.linspace(0.0, z_sn.max(), 1000)   # de hoje ate a supernova mais distante
def log_prior(theta):
    omega_m, omega_lambda = theta
    # fora da caixa tem probabilidade zero
    if not (OM_MIN <= omega_m <= OM_MAX and OL_MIN <= omega_lambda <= OL_MAX):
        return -np.inf
    # sem Big Bang E^2 fica negativo antes da supernova mais distante
    omega_k = 1 - omega_m - omega_lambda
    E2 = omega_m*(1 + z_checagem)**3 + omega_k*(1 + z_checagem)**2 + omega_lambda
    if np.any(E2 <= 0):
        return -np.inf
      # flat
    return 0.0
