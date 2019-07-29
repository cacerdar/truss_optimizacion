from src.nusa import TrussModel
import numpy as np

class FiniteElementLayer(TrussModel):

    def __init__(self, name):
        TrussModel.__init__(self, name="Truss Model 01")
        self._max_displacement = 4
        self._max_displacement_pw = 4**2
        self._max_stress = 30
        self._max_stress_pw = 30**2

    def get_stresses(self):
        stresses = []
        for elm in self.get_elements():
            stresses.append([elm.label+1, elm.s])

        return stresses

    def get_displacements(self):
        displacements = []
        for nodo in self.get_nodes():
            displacements.append([nodo.ux, nodo.uy])

        return displacements

    def isStable(self):
        elems = []

        for elm in self.get_elements():
            if(np.float_power(elm.s,2) > self._max_stress_pw):
                elems.append(elm.label)


        if len(elems) > 0:
            return False, elems

        for nodo in self.get_nodes ():
            if ((np.float_power (nodo.ux , 2) + np.float_power (nodo.uy , 2)) > self._max_displacement_pw):
                return False , elems

        return True, elems