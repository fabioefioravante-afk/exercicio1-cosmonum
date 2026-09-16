#Parte 1 cosmologia


import numpy as np
from scipy.integrate import cumulative_trapezoid


# Todas as distancias abaixo estao em unidades da distancia de Hubble, c/H0 = 2997.92458/h Mpc.
# coloquei um check nas partes que o slide falava pra checar.
# H0= 70 km/s/Mpc, Om=0.3, OL=0.7
C_KMS = 299792.458                      # km/s

class FLRW:
      def __init__(self, H0, omega_m, omega_lambda, omega_r=0.0, z_max=2.0, N=1000):
          self.H0  = H0
          self.Om  = omega_m
          self.OL  = omega_lambda
          self.Or = omega_r
          self.Ok  = 1 - omega_m - omega_lambda- omega_r
          self.z_max = z_max
          self.D_H = C_KMS / H0
          self.zgRADE   = np.linspace(0.0, z_max, N) #divide 0 até z_max em N partes
          self.chi_gRADE = cumulative_trapezoid(1/self.E(self.zgRADE), self.zgRADE, initial=0.0) #cumulative_trapezoid(y(x),x) é a integral de y(x) dx, com o valor inicial da integral = 0.0. 
          #é uma integral aproximada que te dá uma array com os valores da integral de zero até cada valor de zgRADE.
          self.t_H = 977.79222 / H0                 # tempo de Hubble em Gyr
          self.agrade = np.linspace(0.0, 1.0, N)    
          a = self.agrade
          with np.errstate(divide='ignore', invalid='ignore'):
            f = 1/(np.sqrt(self.Or/a**2 + self.Om/a + self.Ok + self.OL*a**2 ))
          f[0] = 0.0                                
          self.tgrade = cumulative_trapezoid(f, a, initial=0.0) #integral do tempo vezes o H0 
          s = np.linspace(0.0, 1.0, N)
          with np.errstate(divide='ignore', invalid='ignore'):
            g = 2*s / np.sqrt(self.Or + self.Om*s**2 + self.Ok*s**4 + self.OL*s**8)
            if self.Or > 0:
                g[0] = 0.0
            else:
                g[0] = 2/np.sqrt(self.Om)
            self.sgrade = s
          self.etagrade = cumulative_trapezoid(g, s, initial=0.0)


      def E(self, z):
          return np.sqrt(self.Or*(1+z)**4 + self.Om*(1+z)**3 + self.Ok*(1+z)**2 + self.OL)
      
      def chi(self, z):
          if np.any(np.asarray(z) > self.z_max):
            raise ValueError(f"z acima de z_max = {self.z_max}")
          return np.interp(z, self.zgRADE, self.chi_gRADE) #faz a função interpolada dos pares ordenados (zgRADE, chi_gRADE) e retorna o valor interpolado de chi para cada z.

      def Sk(self, chi):
          if self.Ok > 1e-8:
              return np.sinh(np.sqrt(self.Ok)*chi)/np.sqrt(self.Ok)
          if self.Ok < -1e-8:
              return np.sin(np.sqrt(-self.Ok)*chi)/np.sqrt(-self.Ok)
          return chi #check

      def D_L(self, z):
          return self.D_H * (1+z) * self.Sk(self.chi(z))

      def mu(self, z):
          return 5*np.log10(self.D_L(z)) + 25

      def D_A(self, z):
          return self.D_H * self.Sk(self.chi(z)) / (1 + z) #check

      def D_M(self, z):
          return self.D_H *self.Sk(self.chi(z)) #check

      def H(self, z):
          return self.H0 * self.E(z)

      def t(self, z):
          return self.t_H * np.interp(1/(1+np.asarray(z)), self.agrade, self.tgrade) #lembra que a=1/(1+z)

      def eta(self, z):
          return self.t_H * np.interp(np.sqrt(1/(1+np.asarray(z))), self.sgrade, self.etagrade) #check

      def dVdzdOmega(self, z):
          return self.D_H * self.D_M(z)**2 / self.E(z)   # Mpc^3 / sr
