import numpy as np
import pytest
from cosmonum.cosmology import FLRW

Z = np.array([0.1, 0.5, 1.0, 2.0])


def test_E_de_hoje_vale_1():
    for Om, OL, Or in [(0.3, 0.7, 0.0), (0.3, 0.9, 0.0), (1.0, 0.0, 0.0), (0.3, 0.7, 9e-5)]:
        assert FLRW(70, Om, OL, omega_r=Or).E(0.0) == pytest.approx(1.0)


def test_lei_de_hubble():
    c = FLRW(70, 0.3, 0.7)
    for z in [1e-3, 1e-4]:
        assert c.D_L(z) / (c.D_H * z) == pytest.approx(1.0, abs=1e-3)


def test_D_M_igual_a_chi_quando_plano():
    for Om, OL in [(0.3, 0.7), (0.2, 0.8)]:
        c = FLRW(70, Om, OL)
        assert np.allclose(c.D_M(Z), c.D_H * c.chi(Z))


def test_D_L_igual_a_D_A_vezes_1_mais_z_ao_quadrado():
    for Om, OL in [(0.3, 0.7), (0.3, 0.9), (0.2, 0.5)]:
        c = FLRW(70, Om, OL)
        assert np.allclose(c.D_L(Z), (1 + Z)**2 * c.D_A(Z))


def test_valores_de_referencia():
    c = FLRW(70, 0.3, 0.7)
    assert np.allclose(c.mu(np.array([0.1, 0.5, 1.0])), [38.3152, 42.2612, 44.1002], atol=1e-3)
    assert c.t(0.0) == pytest.approx(13.4670, abs=1e-3)
    assert c.t(1.0) == pytest.approx(5.7516, abs=1e-3)
    assert c.eta(0.0) == pytest.approx(46.1669, abs=1e-3)
    assert c.eta(1.0) == pytest.approx(35.3912, abs=1e-3)


def test_radiacao():
    c = FLRW(70, 0.3, 0.7 - 9e-5, omega_r=9e-5)
    assert c.Ok == pytest.approx(0.0, abs=1e-12)
    assert c.t(0.0) == pytest.approx(13.4614, abs=1e-3)
    assert c.eta(0.0) == pytest.approx(45.2940, abs=1e-3)


def test_erro_acima_de_z_max():
    c = FLRW(70, 0.3, 0.7)
    with pytest.raises(ValueError):
        c.D_L(3.0)
