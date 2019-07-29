from abc import abstractmethod
from enum import Enum
import numpy as np
import random

class ENFRIAMIENTO(Enum):
    GEOM = 1
    LOG = 2
    LIN = 3


class SimulatedAnnealingMethod():

    def __init__(self, tmax, iteraciones=100, enfriamiento=ENFRIAMIENTO.LIN, coeficiente=1):
        self._temp = tmax
        self._tmin = tmax*0.01
        self._fitness = []
        self._kbs = []
        self.kbs = 5889.99
        self._solucion = []
        self._iteraciones = iteraciones
        self._tipo_enfriamiento = enfriamiento
        self._soluciones = []
        self._b = coeficiente  #coef. enfriamiento geométrico


    def set_solucion_inicial(self, solucion):
        self._solucion = solucion


    def run(self):

        while int(self._tmin) < int(self._temp):  # and self._temp >= 1:
            itera = self._iteraciones
            while itera > 0:

                solucion1 = self.construye_vecindad(self._solucion)
                fitness_solucion1 = self.evaluar(solucion1)

                fitness_solucion  = self.evaluar(self._solucion)

                de = fitness_solucion1 - fitness_solucion

                if de <= 0:
                    self._solucion = solucion1
                    fitness_solucion = fitness_solucion1
                else:
                    prob = np.exp(-de/self._temp)
                    rand = random.uniform(0, 1)
                    if prob > rand:
                        #print("acepta peor solucion")
                        self._solucion = solucion1
                        fitness_solucion = fitness_solucion1

                itera -= 1
                self._fitness.append(fitness_solucion)
                self._soluciones.append(self._solucion)
                self._kbs.append((fitness_solucion - self.kbs) / self.kbs)


            if self._tipo_enfriamiento is ENFRIAMIENTO.GEOM:
                    self._temp = self._b * self._temp
            elif self._tipo_enfriamiento is ENFRIAMIENTO.LIN:
                    self._temp -= self._b


    def get_fitness(self):
        return self._fitness

    def get_kbs(self):
        return self._kbs

    def get_soluciones(self):
        return self._soluciones

    @abstractmethod
    def construye_vecindad(self, solucion):
        pass

    @abstractmethod
    def construye_vecindad(self, solucion, perfiles):
        pass

    @abstractmethod
    def evaluar(self, solucion, f, m):
        pass