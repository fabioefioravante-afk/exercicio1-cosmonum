import numpy as np

# Parte 2 estatística

class Posterior:
    def __init__(self, model, log_likelihood, log_prior):
        self.model = model                    # a cosmologia: a classe FLRW la de cima
        self.log_likelihood = log_likelihood  # funcao: o quanto os dados casam com theta
        self.log_prior = log_prior            # funcao: o que eu assumo antes de ver os dados

    def __call__(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf     #mata tudo fora da priori
        return lp + self.log_likelihood(theta, self.model)


class Proposal:
    def propose(self, theta, rng):
        raise NotImplementedError
    def logpdf(self, theta_to, theta_from):
        raise NotImplementedError



class GaussianRandomWalk(Proposal):
    def __init__(self, cov):
        self.cov = np.atleast_2d(np.asarray(cov, float)) # garantir que seja uma matriz 2D
        self.L = np.linalg.cholesky(self.cov)
        self.d = self.cov.shape[0] 
        self. logdet = np.log(np.diag(self.L)).sum() 

    def propose(self, theta, rng):
        z = rng.normal(size=self.d) #sorteia ruido redondo
        return theta + self.L @ z    #transforma em elipse e soma ao ponto atual
    def logpdf(self, theta_to, theta_from):
        # calcula o log da probabilidade de ir de theta_from para theta_to
        y = np.linalg.solve(self.L, np.asarray(theta_to) - np.asarray(theta_from))
        return -0.5*(y @ y) - self.logdet - 0.5*self.d*np.log(2*np.pi)


class MetropolisHastings:
    
    def __init__(self, log_target, proposal):
        self.log_target = log_target #funcao que da o log da posterior (a peca 1)
        self.proposal = proposal #o objeto que sabe sortear e avaliar pulos (a peca 2)

    def sample(self, theta0, n_passos, rng):
        # theta0   = ponto de onde a cadeia comeca
        # n_passos = quantas iteracoes rodar
        # rng      = o sorteador de numeros aleatorios, vindo DE FORA. E ele que
        #            faz duas rodadas com a mesma semente darem exatamente o
        #            mesmo resultado (um dos testes que o exercicio pede). Se o
        #            codigo chamasse np.random.normal direto aqui dentro,
        #            estaria usando um sorteador global escondido e a
        #            reprodutibilidade iria embora.
        estado = np.asarray(theta0, float)
        logp = self.log_target(estado)
        if not np.isfinite(logp):
            raise ValueError("estado inicial impossível")

        amostras = np.empty((n_passos, len(estado))) #tabela: uma linha por passo, uma coluna por parametro
        logps = np.empty(n_passos) #um numero por passo
        aceito = np.zeros(n_passos, dtype=bool)

        for i in range(n_passos):
            candidato = self.proposal.propose(estado, rng) #gera um candidato a partir do estado atual
            logp_novo = self.log_target(candidato) #calcula o log da posterior


            log_alpha = logp_novo - logp 
            log_alpha += self.proposal.logpdf(estado, candidato)
            log_alpha -= self.proposal.logpdf(candidato, estado)

            if np.isfinite(log_alpha) and np.log(rng.random()) < min(0.0, log_alpha):
                  # So aqui o logp e atualizado -- e o cache mencionado la em cima.
                  estado, logp = candidato, logp_novo
                  aceito[i] = True

            amostras[i] = estado
            logps[i]    = logp

        return amostras, logps, aceito, aceito.mean()
