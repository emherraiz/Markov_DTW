'''
Utilizaremos este modulo para hacer un repaso del bayesiano

Returns:
    float: Probabilidad
'''
from scipy.stats import norm
class Normal:

    def __init__(self, mu, sigma):
        '''Este es el constructor

        Args:
            mu (float): Valor esperado (media)
            sigma (float): Varianza
        '''
        self.mu = mu
        self.sigma = sigma


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


