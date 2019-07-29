
import numpy as np
import random
import time

import matplotlib.pyplot as plt
from src.FiniteElementLayer import FiniteElementLayer
from src.metaheuristics.SimulatedAnnealing import SimulatedAnnealingMethod as sam
from src.metaheuristics.SimulatedAnnealing import ENFRIAMIENTO
from src.nusa import Node, Truss



class TrussOptimization(sam):

    def __init__(self, temp_max, itera, tipo=ENFRIAMIENTO.LIN, coeficiente=1):
        self._fitness_count = 0          # Starts the fitness count at this value
        self._fitness_count_limit = 80   # How many times the fitness can remain unchanged.
        self._fireup_count = 1           #  N times that the temperature will be raised up
        self._fireup = temp_max * 0.6    # How high the temperature will be raised up at fireup execution
        self._fireup_threshold = temp_max * 0.40 # At what temperature the fireup action will be executed

        sam.__init__(self , temp_max , itera , tipo , coeficiente)
        self._tmin = 0.1  # Lowest cooling temperature. After reaching this value the process will be stopped.
        # List of all the possible section areas.
        self._A = [(float("{0:.2f}".format(x))) for x in np.arange(0.1, 15.1, 0.1)]
        # Instantiates the Truss
        self.btruss = BuildTruss()
        #The process starts with a random solution
        solucion = [(self._A[int(random.uniform(0, 149))]) for x in range(1, 39)]
        #_largo: how many of the bars will be changed for each neighbour
        self._largo = int(len(solucion) / 8)  # 1/8 del vector solucion
        self.set_solucion_inicial(solucion)


    # Check if determinated element exist in the parameter: list_element.
    # Returns: True: if exist in list
    #          False: Doesn't exist in the list
    def is_element_in_list(self, element, list_element):
        try:
            list_element.index(element)
            return True
        except ValueError:
            return False

    # Function: construye_vecindad
    #       Generates one new neighbour from the solucion vector.
    # Parameters: solucion. Best solution found in the present cycles.
    # Returns: returns a new neighbour, candidate to be the new best found solution.
    def construye_vecindad(self, solucion):
        elems = []
        largo_areas = len(self._A)

        vecino = solucion.copy()

        # repeat until found one stable neighbour
        while True:

            if self._temp < self._fireup_threshold and self._fitness_count > self._fitness_count_limit and self._fireup_count > 0:
                self._tipo_enfriamiento = ENFRIAMIENTO.LOG
                self._b = 10.3  # Coefficient for Logarithmic cool down
                self._fireup_count -= 1
                self._largo = 1 # From now on the neighbour will be accept only 1 change of its members.

            # change the section of the rejected elements. This speedup at least a 90% the overall process !
            if len(elems) > 0:
                for e in elems:
                    rand_index = int(random.uniform(self._A.index(vecino[e]), int(largo_areas)))
                    vecino[e] = self._A[rand_index]

            # change randomly  _largo qty elements from the neighbour
            for i in range(self._largo):
                rand_index = int(random.uniform(0, int(largo_areas)))
                rand_index2 = int(random.uniform(0, len(solucion)))
                vecino[rand_index2] = self._A[rand_index]

            #Check if the new neighbour check for displacement and stress restrictions.
            #elems is a list of the rejected elements.
            is_stable, elems = self.btruss.is_stable(vecino)

            # if it is stable, the neighbour it is a candidate to be the new best solution, for hence,
            # return this new neighbour. Else, continue until found one stable neighbour
            if is_stable:
                lfitness = len(self.get_fitness())
                if lfitness > 2 and self.get_fitness()[lfitness-1] == self.get_fitness()[lfitness - 2]:
                    self._fitness_count += 1
                else:
                    self._fitness_count = 0

                return vecino


    # Function: evaluar
    # Evaluates the found solution in the Objective Function
    # Parameters:
    #       solucion: best solution found at each temperature
    # Returns the Fitness of the solution.
    def evaluar(self, solucion):
        fitness = 0
        largo = len(solucion)
        p = 0.283  # lbf/in3  peso especifico del acero
        #gets the Length of each bar
        f = self._build_L_vector()
        for i in range(largo):
            fitness = fitness + solucion[i] * f[i] * p

        return fitness

    def _build_L_vector(self):
        L = []
        elem = self.btruss.truss.get_elements()
        for el in elem:
            L.append(el.get_length())

        return L


class BuildTruss():

    def __init__(self):
        pass

    # Function: _add_barras:
    #  add elements with A section to re-create the truss.
    #  Parameters:
    #           nodes_up: nodes from the upper side of the truss
    #           nodes_down: nodes from the bottom side of the truss
    #           section_area: section area for each new bar element.
    #  Return: list of bar elements.
    def _add_barras(self , nodes_up , nodes_down , section_area):

        # Elementos barra
        bar_index = 0
        barra = [ (Truss((nodes_up[x - 1], nodes_up[x]) , self.E , section_area[ bar_index + x - 1 ])) for x in
                  range(1, 10) ]  # barras superiores del 0 al 8
        bar_index += 9
        # barras inferiores del 9 al 18
        barra.extend(([ (Truss((nodes_down[x - 1], nodes_down[x]) , self.E , section_area[ bar_index + x - 1 ])) for x in
                        range(1, 11) ]))
        bar_index += 9
        # barras verticales
        barra.extend(([ (Truss((nodes_up[x], nodes_down[x]) , self.E , section_area[ bar_index + x - 1 ])) for x in
                        range(1, 10) ]))
        bar_index += 9
        # barras diagonales
        barra.extend(([ (Truss((nodes_up[x - 1], nodes_down[x]) , self.E , section_area[ bar_index + x - 1 ])) for x in
                        range(1, 11) ]))

        return barra

    # Function: is_stable:
    #  Checks if the re-created truss is mechanically stable and if it's ok with maximum stress and displacement.
    #  Parameters:
    #           section_area: section area for each new bar element.
    #  Return: list of bar elements.
    def is_stable(self , section_area):

        self.truss = FiniteElementLayer("Truss Model")
        self.E = 30000  # ksi. Young Modulus. Material: steel.
        L = 100         # inches. Length of each horizontal-vertical element.

        self.nodes_up = [(Node((L * x, L))) for x in range(0, 10)]  # Nodos 0 al 9
        self.nodes_down = [(Node((L * x, 0))) for x in range(0, 11)]  # Nodos 0 al 10

        # Elementos barra
        barra = self._add_barras(self.nodes_up , self.nodes_down , section_area)

        for nd in self.nodes_up:
            self.truss.add_node(nd)

        for nd in self.nodes_down:
            self.truss.add_node(nd)

        for el in barra:
            self.truss.add_element(el)

        # Restriccion de desplazamiento en nodos
        self.truss.add_constraint(self.nodes_up[0], ux=0, uy=0)
        self.truss.add_constraint(self.nodes_down[0], ux=0, uy=0)
        self.truss.add_force(self.nodes_down[10], (0, -15))

        self.truss.solve()

        return self.truss.isStable()



class Main():

    def run(self , filename , fignumber , maxtemp , iteracion , tipo_enfriamiento , coeficiente):
        # creates the Truss
        t = TrussOptimization(maxtemp , iteracion , tipo_enfriamiento , coeficiente)

        # Starts the metaheuristic
        t.run()
        #get the fitness vector after reaching a solution
        fitness = t.get_fitness()
        #get the solutions vector obtained at each temperature
        soluciones = t.get_soluciones()
        #get the relation of each solution regarding the known best solution.
        kbs = t.get_kbs()

        #Save the results
        plt.subplot(2, 1, 1)
        plt.plot(fitness)
        plt.ylabel('Fitness')

        plt.subplot(2, 1, 2)
        plt.plot(kbs)
        plt.ylabel('KBS')

        plt.savefig(fignumber)

        with open("..\\" + filename + ".kbs", "w", newline="") as f:
            f.write("\n".join(str(line) for line in kbs))

        with open("..\\" + filename + ".fit", "w", newline="") as f:
            f.write("\n".join(str(line) for line in fitness))

        with open("..\\" + filename + ".sol", "w", newline="") as f:
            f.write("\n".join(str(line) for line in soluciones))

        plt.clf()
        plt.close()


#Begin of my Metaheuristic
m = Main()

#nestabilidad: Controls in how many cycles the temperature will be considered stabilized.
nestabilidad = 20 # Half of the neighbours vector size. Controls how much the neighbours will be changed.
t_init = 600 # Initial Temperature value.
cooling_coefficient = 0.7  # Decremental coefficient. 70% of the last temperature in each cycle.

#Run the Heuristic 10 times.
for i in range(1, 11):
    print("Iteration " + str(i))
    figname = "Fig." + str(i) + ".png"
    filename = "Fig" + str(i) + ".txt"
    m.run(filename , figname , t_init , nestabilidad , ENFRIAMIENTO.GEOM , cooling_coefficient)
