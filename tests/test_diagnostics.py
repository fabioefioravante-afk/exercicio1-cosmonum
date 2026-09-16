import numpy as np
from cosmonum.diagnostics import (taxa_movel, autocorrelacao, tempo_autocorrelacao,
                                  ess, mcse, split_rhat, dunkley)


def ar1(phi, n, rng):
    # serie em que cada valor e phi vezes o anterior mais um ruido; tau exato = (1+phi)/(1-phi)
    x = np.empty(n)
    x[0] = rng.normal()
    ruido = rng.normal(size=n) * np.sqrt(1 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + ruido[t]
    return x


def test_taxa_movel():
    assert np.allclose(taxa_movel(np.ones(5_000, dtype=bool)), 1.0)
    assert np.allclose(taxa_movel(np.tile([True, False], 2_500)), 0.5)


def test_autocorrelacao_e_tau_do_ar1():
    x = ar1(0.9, 200_000, np.random.default_rng(1))
    assert np.allclose(autocorrelacao(x, 5)[1:], 0.9 ** np.arange(1, 6), atol=0.02)
    assert abs(tempo_autocorrelacao(x) / 19 - 1) < 0.15
    assert abs(ess([x]) / (200_000 / 19) - 1) < 0.15


def test_tau_do_ruido_branco():
    x = np.random.default_rng(2).normal(size=100_000)
    assert abs(tempo_autocorrelacao(x) - 1) < 0.1


def test_mcse_preve_o_espalhamento_das_medias():
    rng = np.random.default_rng(3)
    cadeias = [ar1(0.9, 5_000, rng) for _ in range(400)]
    real = np.std([x.mean() for x in cadeias])
    previsto = np.mean([mcse([x]) for x in cadeias])
    assert abs(real / previsto - 1) < 0.1


def test_split_rhat():
    iid = np.random.default_rng(4).normal(size=(4, 10_000))
    assert split_rhat(iid) < 1.01
    deslocada = iid.copy()
    deslocada[0] += 1.0
    assert split_rhat(deslocada) > 1.05
    assert split_rhat(iid + np.linspace(0, 2, 10_000)) > 1.05


def test_dunkley_ruido_branco():
    res = dunkley(np.random.default_rng(5).normal(scale=2.0, size=50_000))
    assert abs(res["P0"] / 4.0 - 1) < 0.1
    assert res["passa"]


def test_dunkley_ar1():
    phi, n = 0.9, 50_000
    res = dunkley(ar1(phi, n, np.random.default_rng(6)))
    j_star_esperado = (1 - phi) / np.sqrt(phi) * n / (2 * np.pi)
    assert abs(res["P0"] / 19 - 1) < 0.2
    assert abs(res["alpha"] - 2) < 0.3
    assert abs(res["j_star"] / j_star_esperado - 1) < 0.25


def test_dunkley_cadeia_lenta_nao_passa():
    res = dunkley(ar1(0.999, 20_000, np.random.default_rng(7)))
    assert not res["passa"]
