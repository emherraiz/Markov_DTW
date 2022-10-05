import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  scipy.stats import norm

def estado(v_inicial, m_transicion):
    v_estado = [v_inicial]

    for i in range(50):
        v_estado.append(v_estado[-1] @ m_transicion)


    return pd.DataFrame(v_estado)

m_transicion = np.array([[.6, .2, .8],[.05, .9, .05],[.1, .2, .7]])
v_inicial = np.array([.1, .8, .1])

def equilibrium_distribution(p_transition):
    """This implementation comes from Colin Carroll, who kindly reviewed the notebook"""
    # Shape nos devuelve una lista con la dimensón de la matriz
    n_states = p_transition.shape[0]

    # eye nos da una matriz diagonal de unos
    # Reshape nos hace una matriz primer_parametro * segundo_parametro * ... * n_parametro
    A = np.append(
        arr=p_transition.T - np.eye(n_states),
        values=np.ones(n_states).reshape(1, -1),
        axis=0
    )

    # lingalgmpinv nos devuelve la psudoinversa: (A^T * A)^{-1} * A^T
    pinv = np.linalg.pinv(A)

    # Devuelve la última columna
    return pinv.T[-1]


print(equilibrium_distribution(m_transicion))

# Funcion de densidad

class Normal:
    def __init__(self, mu, sigma):
        # Valor esperado (media)
        self.mu = mu
        # Varianza
        self.sigma = sigma
        # NOTA: no necesitamos de la

        # Instanciamos un objeto con la distribución normal
        self.dist = norm(loc = mu, scale = sigma)

    # Funcion de densidad de probabilidad (gaussiana) a la cual le introducimos un registro
    # La integral en el rango especificado siempre es = 1
    # La probabilidad debe estar en el rango [0  1]
    def pdf(self, x):
        # (1 / np.sqrt(2 * self.sigma ** 2 * np.pi) * np.exp( - (x - self.mu) ** 2 / 2 * self.sigma ** 2))
        return self.dist.pdf(x)

    # Nos quedamos con las probabilidades logaritmicas de cada uno de los registros
    def logpdf(self, x):
        # Sumatorio_xi(log(pdf(xi)))
        return self.dist.logpdf(x)

    # Generamos una lista de n probabilidades aleatorias sujetas a la funcion de probabilidad
    def sample(self, n):
        return self.dist.rvs(n)

print(norm(loc=0, scale=1).rvs(10))


