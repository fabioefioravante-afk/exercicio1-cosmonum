import numpy as np
import pytest
from scipy.stats import multivariate_normal
from cosmonum.bayes import GaussianRandomWalk, MetropolisHastings

MU = np.array([1.0, -2.0])
COV = np.array([[4.0, 1.8], [1.8, 1.0]])
L = np.linalg.cholesky(COV)


def log_alvo(x):
    y = np.linalg.solve(L, x - MU)
    return -0.5 * (y @ y)


def roda(semente, n_passos):
    mh = MetropolisHastings(log_alvo, GaussianRandomWalk(2.38**2 / 2 * COV))
    return mh.sample([0.0, 0.0], n_passos, np.random.default_rng(semente))


def test_recupera_gaussiana_conhecida():
    amostras, logps, aceito, taxa = roda(20260906, 200_000)
    depois = amostras[20_000:]
    assert 0.25 < taxa < 0.45
    assert np.allclose(depois.mean(axis=0), MU, atol=0.05)
    assert np.allclose(np.cov(depois.T), COV, atol=0.1)


def test_mesma_semente_mesmo_resultado():
    a, logps_a, aceito_a, _ = roda(42, 5_000)
    b, logps_b, aceito_b, _ = roda(42, 5_000)
    c, _, _, _ = roda(43, 5_000)
    assert np.array_equal(a, b)
    assert np.array_equal(logps_a, logps_b)
    assert np.array_equal(aceito_a, aceito_b)
    assert not np.array_equal(a, c)


def test_rejeitado_repete_o_estado():
    amostras, logps, aceito, taxa = roda(7, 5_000)
    rej = ~aceito[1:]
    assert np.all(amostras[1:][rej] == amostras[:-1][rej])
    assert np.all(logps[1:][rej] == logps[:-1][rej])
    assert np.all(amostras[1:][~rej] != amostras[:-1][~rej])
    assert taxa == aceito.mean()


def test_logpdf_da_proposta():
    prop = GaussianRandomWalk(COV)
    de, para = np.array([0.3, -1.0]), np.array([1.2, 0.4])
    assert prop.logpdf(para, de) == pytest.approx(multivariate_normal(de, COV).logpdf(para))
