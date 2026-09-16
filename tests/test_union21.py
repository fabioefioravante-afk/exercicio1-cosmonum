import numpy as np
import pytest
from scipy.linalg import solve_triangular
from cosmonum.cosmology import FLRW
from cosmonum.bayes import Posterior
from cosmonum.union21 import (N_SN, z_sn, mu_obs, sigma_mu, cov_sys, L_sys,
                              carrega_covariancia, chi2_marg, log_likelihood, log_prior)

MU_REF = FLRW(70, 0.29, 0.68).mu(z_sn)


def test_dimensoes_dos_dados():
    assert z_sn.shape == mu_obs.shape == sigma_mu.shape == (N_SN,)
    assert cov_sys.shape == (N_SN, N_SN)
    assert np.allclose(cov_sys, cov_sys.T)
    assert np.allclose(L_sys @ L_sys.T, cov_sys)
    assert z_sn.max() < FLRW(70, 0.3, 0.7).z_max


def test_diagonal_sem_sistematicas_ja_e_sigma_mu_ao_quadrado():
    assert np.allclose(np.diag(carrega_covariancia(sistematicas=False)), sigma_mu**2)


def test_chi2_por_cholesky_igual_a_conta_direta():
    delta = mu_obs - MU_REF
    uns = np.ones(N_SN)
    C_delta = np.linalg.solve(cov_sys, delta)
    C_uns = np.linalg.solve(cov_sys, uns)
    esperado = delta @ C_delta - (delta @ C_uns)**2 / (uns @ C_uns)
    assert chi2_marg(MU_REF) == pytest.approx(esperado)


def test_chi2_nao_depende_de_M_nem_de_H0():
    assert chi2_marg(MU_REF + 0.5) == pytest.approx(chi2_marg(MU_REF))
    assert chi2_marg(FLRW(50, 0.29, 0.68).mu(z_sn)) == pytest.approx(chi2_marg(MU_REF))


def test_chi2_marg_e_o_minimo_em_M():
    delta = mu_obs - MU_REF
    varredura = []
    for M in np.linspace(-0.2, 0.2, 401):
        r = solve_triangular(L_sys, delta - M, lower=True)
        varredura.append(r @ r)
    menor = min(varredura)
    assert chi2_marg(MU_REF) <= menor + 1e-9
    assert menor - chi2_marg(MU_REF) < 0.01


def test_melhor_ajuste():
    assert chi2_marg(MU_REF) == pytest.approx(545.13, abs=0.01)


def test_log_likelihood():
    assert log_likelihood((0.29, 0.68), FLRW) == pytest.approx(-0.5 * chi2_marg(MU_REF))
    assert log_likelihood((0.3, 1.7), FLRW) == -np.inf


def test_priori():
    assert log_prior((0.3, 0.7)) == 0.0
    for fora in [(-0.1, 0.7), (2.1, 0.7), (0.3, -1.1), (0.3, 3.1)]:
        assert log_prior(fora) == -np.inf
    assert log_prior((0.1, 2.0)) == -np.inf
    assert log_prior((0.3, 1.7)) == 0.0


def test_posterior():
    post = Posterior(FLRW, log_likelihood, log_prior)
    assert np.isfinite(post((0.29, 0.68)))
    for impossivel in [(3.0, 0.5), (0.1, 2.0), (0.3, 1.7)]:
        assert post(impossivel) == -np.inf
