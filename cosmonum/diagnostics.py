import numpy as np
from scipy.optimize import curve_fit


def taxa_movel(aceitos, janela=1_000):
    # fracao de passos aceitos em cada janela de passos seguidos
    return np.convolve(aceitos, np.ones(janela) / janela, mode="valid")

def autocorrelacao(x, max_lag):
    # rho[l]: o quanto a amostra i se parece com a amostra i+l
    x = np.asarray(x, float) - np.mean(x)
    soma_total = x @ x
    return np.array([x[:len(x) - l] @ x[l:] / soma_total for l in range(max_lag + 1)])

def tempo_autocorrelacao(x, c=5):
    # tau = 1 + 2*(rho_1 + rho_2 + ...), somando ate a janela passar de c*tau
    x = np.asarray(x, float) - np.mean(x)
    soma_total = x @ x
    tau = 1.0
    for l in range(1, len(x)):
        tau += 2 * (x[:len(x) - l] @ x[l:]) / soma_total
        if l >= c * tau:
            return tau
    raise ValueError("a cadeia e curta demais para estimar tau")

def ess(cadeias_1param):
    # amostras independentes somadas sobre as cadeias
    return sum(len(x) / tempo_autocorrelacao(x) for x in cadeias_1param)

def mcse(cadeias_1param):
    # erro na media causado pelo tamanho finito das cadeias
    return np.std(cadeias_1param) / np.sqrt(ess(cadeias_1param))

def split_rhat(cadeias_1param):
    # corta cada cadeia ao meio e compara a variancia dentro dos pedacos com a de todos juntos
    cadeias_1param = np.asarray(cadeias_1param, float)
    n = cadeias_1param.shape[1] // 2
    pedacos = np.concatenate([cadeias_1param[:, :n], cadeias_1param[:, n:2 * n]])   # (2*N_cadeias, n)
    medias = pedacos.mean(axis=1)
    B = n * np.var(medias, ddof=1)                    # variacao entre os pedacos
    W = np.mean(np.var(pedacos, axis=1, ddof=1))      # variacao dentro de cada pedaco
    V = (n - 1) / n * W + B / n                       # variancia de tudo junto
    return np.sqrt(V / W)

def espectro(x):
    # P_j = |a_j|^2, com a_j = soma de (x_n - media) e^{2 pi i j n / N} / sqrt(N)
    x = np.asarray(x, float) - np.mean(x)
    N = len(x)
    a = np.fft.fft(x) / np.sqrt(N)
    j = np.arange(1, N // 2)
    return j, np.abs(a[j])**2

def modelo_log_espectro(j, ln_P0, ln_jstar, alpha):
    # log de P0 / (1 + (j/j*)^alpha), menos a constante de Euler (Dunkley et al. 2005, eq. 19)
    return ln_P0 - np.log1p((j / np.exp(ln_jstar))**alpha) - np.euler_gamma

def dunkley(x):
    x = np.asarray(x, float)
    N = len(x)
    j, P = espectro(x)
    j_max = 1_000                                     # primeira passada, como no artigo
    for passada in range(2):
        usar = j <= j_max
        chute = [np.log(P[:10].mean()), np.log(50.0), 2.0]
        limites = ([-np.inf, 0.0, 1.0], [np.inf, np.log(100 * N), 6.0])
        (ln_P0, ln_jstar, alpha), _ = curve_fit(modelo_log_espectro, j[usar], np.log(P[usar]),
                                                 p0=chute, bounds=limites)
        j_max = min(10 * np.exp(ln_jstar), N // 2 - 1)  # segunda passada: j_max = 10 j*
    P0, j_star = np.exp(ln_P0), np.exp(ln_jstar)
    r = P0 / (N * np.var(x))
    return {"P0": P0, "j_star": j_star, "alpha": alpha, "r": r,
              "passa": bool(j_star > 20 and r < 0.01)}
