import random as rd

PRICE = {
    "H": 1.39, "He": 24, "Li": 85.6, "Be": 857, "B": 3.68, "C": 0.122,
    "N": 0.14, "O": 0.154, "F": 2.16, "Ne": 240, "Na": 3.43, "Mg": 2.32,
    "Al": 1.79, "Si": 1.7, "P": 2.69, "S": 0.0926, "Cl": 0.082, "Ar": 0.931,
    "K": 13.6, "Ca": 2.35, "Sc": 3460, "Ti": 11.7, "V": 385, "Cr": 9.4, "Mn": 1.82,
    "Fe": 0.424, "Co": 32.8, "Ni": 13.9, "Cu": 6, "Zn": 2.55, "Ga": 1480,
    "Ge": 1010, "As": 1.31, "Se": 21.4, "Br": 4.39, "Kr": 290, "Rb": 15500, "Sr": 6.68,
    "Y": 31, "Nb": 85.6, "Mo": 40.1, "Tc": 100000, "Ru": 10600, "Rh": 147000, "Pd": 49500, "Ag": 521,
    "Cd": 2.73, "In": 167, "Sn": 18.7, "Sb": 5.79, "Te": 63.5, "I": 35, "Xe": 1800,
    "Cs": 61800, "Ba": 275, "La": 4920, "Ce": 4710, "Pr": 103000, "Nd": 57500, "Pm": 460000,
    "Sm": 13900, "Eu": 31400, "Gd": 28600, "Tb": 658000, "Dy": 307000, "Ho": 57100, "Er": 26400,
    "Tm": 3000, "Yb": 17100, "Lu": 643000, "Hf": 900, "Ta": 312000, "W": 35300, "Re": 4150, "Os": 12000,
    "Ir": 56200, "Pt": 27800, "Hg": 30200, "Tl": 4200, "Pb": 2000, "Bi": 6360, "Po": 49200000000,
    "Ac": 29000000000, "Th": 287, "Pa": 280000, "U": 101, "Np": 660000, "Pu": 6490000, "Am": 750000,
    "Cm": 160000000, "Bk": 185000000, "Cf": 185000000
}

ATOMIC_WEIGHT = {
    "H": 1.008, "He": 4.002602, "Li": 6.94, "Be": 9.0121831, "B": 10.81, "C": 12.011,
    "N": 14.007, "O": 15.999, "F": 18.998403163, "Ne": 20.1797, "Na": 22.98976928,
    "Mg": 24.305, "Al": 26.9815385, "Si": 28.085, "P": 30.973761998, "S": 32.06,
    "Cl": 35.45, "Ar": 39.948, "K": 39.0983, "Ca": 40.078, "Sc": 44.955908,
    "Ti": 47.867, "V": 50.9415, "Cr": 51.9961, "Mn": 54.938044, "Fe": 55.845,
    "Co": 58.933194, "Ni": 58.6934, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723,
    "Ge": 72.63, "As": 74.921595, "Se": 78.971, "Br": 79.904, "Kr": 83.798,
    "Rb": 85.4678, "Sr": 87.62, "Y": 88.90584, "Nb": 92.90637, "Mo": 95.95,
    "Tc": 97.90721, "Ru": 101.07, "Rh": 102.9055, "Pd": 106.42, "Ag": 107.8682,
    "Cd": 112.414, "In": 114.818, "Sn": 118.71, "Sb": 121.76, "Te": 127.6,
    "I": 126.90447, "Xe": 131.293, "Cs": 132.90545196, "Ba": 137.327,
    "La": 138.90547, "Ce": 140.116, "Pr": 140.90766, "Nd": 144.242,
    "Pm": 144.91276, "Sm": 150.36, "Eu": 151.964, "Gd": 157.25, "Tb": 158.92535,
    "Dy": 162.5, "Ho": 164.93033, "Er": 167.259, "Tm": 168.93422, "Yb": 173.045,
    "Lu": 174.9668, "Hf": 178.49, "Ta": 180.94788, "W": 183.84, "Re": 186.207,
    "Os": 190.23, "Ir": 192.217, "Pt": 195.084, "Hg": 200.592, "Tl": 204.38,
    "Pb": 207.2, "Bi": 208.9804, "Po": 209.0, "Ac": 227.0, "Th": 232.0377,
    "Pa": 231.03588, "U": 238.02891, "Np": 237.0, "Pu": 244.0, "Am": 243.0,
    "Cm": 247.0, "Bk": 247.0, "Cf": 251.0
}

ELEMENTS = list(PRICE.keys())

def individual_cp(num_elements):
    '''Creates a list containing a list with the elements and another one with the weights.
    
    Args:
      num_elements: number of elements in the compound.
      
    Returns:
      individual: list representing the compound.
      '''    
    elements = rd.sample(ELEMENTS, num_elements)
    weights = []
    weights.append(rd.uniform(5, 90))
    
    for _ in range(num_elements - 2):
        weight = rd.uniform(5, 95 - sum(weights))
        weights.append(weight)
        
    weights.append(100 - sum(weights))
    
    individual = [elements, weights]
    
    return individual

def population_cp(population_size, num_elements):
    '''Creates a population of compounds.
    
    Args:
      population_size: number of individuals in the population.
      
      num_elements: number of elements in the compound.
      
    Returns:
      population: list of individuals.
    '''
    population = []
    
    for _ in range(population_size):
        individual = individual_cp(num_elements)
        population.append(individual)
        
    return population
        
def individual_fitness_cp(individual):
    '''Calculates the weight and price of the compound. It returns price - weight, which needs to be maximized.
    
    Args:
      individual: coumpound.
      
    Returns:
      fitness: list with the weight and price of the compound.
    '''
    elements, weights = individual[0], individual[1] 
    weight, price = 0, 0
    
    for i in range(len(elements)):
        weight += ATOMIC_WEIGHT[elements[i]] * weights[i]
        price += PRICE[elements[i]] * weights[i]
        
    maximum_weight = 90 * ATOMIC_WEIGHT['Cf'] + 5 * ATOMIC_WEIGHT['Bk'] + 5 * ATOMIC_WEIGHT['Cm']
    maximum_price = 90 * PRICE['Po'] + 5 * PRICE['Ac'] + 5 * PRICE['Bk']
    
    fitness = - (weight / maximum_weight) + (price / maximum_price)
    return fitness
            
def population_fitness_cp(population):
    '''Calculates the fitness for a population.
    
    Args:
      population: list of individuals.
            
    Returns:
      population_fitness: list with all fitness values from the individuals in the population.
      
    '''
    population_fitness = []
    
    for individual in population: 
        fitness = individual_fitness_cp(individual)
        population_fitness.append(fitness)
        
    return population_fitness

#################################################################################    
############################## MUTATION OPERATORS ###############################
#################################################################################
        
def element_mutation_cp(individual, individual_mutation_rate=0.05, element_mutation_rate=0.25):
    '''Mutates one or more element from a individual, with a certain mutation rate.
    
    Args:
      individual: list representing a solution.
      
      mutation_rate: value between 0 and 1.
      
    Returns:
      individual: list representing a solution.
    '''
    value = rd.random()
    num_elements = len(individual[0])
    
    if value < individual_mutation_rate:
        for i in range(num_elements):
            value = rd.random()
            
            if value < element_mutation_rate:
                individual[0][i] = rd.choice(ELEMENTS)
        
    return individual

def population_element_mutation_cp(population, individual_mutation_rate=0.05, element_mutation_rate=0.25):
    '''Applies the element mutation to a whole population:
    
    Args:
      population: list of individuals.
      
      mutation_rate: value between 0 and 1.
      
    Returns:
      population: list of individuals.
    
    '''    
    for individual in population:
        mutated_individual = element_mutation_cp(individual, individual_mutation_rate, element_mutation_rate)    
    
    return population
        
def weight_mutation_cp(individual, mutation_rate=0.05):
    '''Mutates the weight of an individual, with a certain mutation rate.
    
    Args:
      individual: possible solution.
            
      mutation_rate: value between 0 and 1.
      
    Returns:
      individual: list representing a solution.
    '''
    value = rd.random()
    weights = individual[1]
    num_elements = len(weights)
    
    if value < mutation_rate:
        index = rd.sample([i for i in range(num_elements)], num_elements - 1)
        bound = min(90 - weights[index[0]], 90 - weights[index[1]], weights[index[0]] - 5, weights[index[1]] - 5)
        weight_added = rd.uniform(-bound, bound) 
        weights[index[0]] += weight_added 
        weights[index[1]] -= weight_added 
        individual[1] = weights
        
    return individual
        
def population_weight_mutation_cp(population, mutation_rate=0.05):
    '''Applies the weight mutation to a whole population:
    
    Args:
      population: list of individuals.
            
      mutation_rate: value between 0 and 1.
      
    Returns:
      population: list of individuals.
    
    '''    
    for individual in population:
        mutated_individual = weight_mutation_cp(individual, mutation_rate)
        
    return population
        
##################################################################################    
############################### CROSSOVER OPERATORS ##############################
##################################################################################

def pair_crossover_cp(parent1, parent2, crossover_rate=0.5):
    '''Crossover operator that recombines the weights and elements from two solutions.
    
    Args:
      parent1, parent2: individuals representing solutions.
      
      crossover_rate: value between 0 and 1.
      
    Returns:
      individual1, individual2: individuals representing solutions.
    '''
    value = rd.random()
    
    if value < crossover_rate:
        individual1 = [parent1[0], parent2[1]]
        individual2 = [parent2[0], parent1[1]]
        return individual1, individual2
    else:
        return parent1, parent2
        
def population_pair_crossover_cp(population, crossover_rate=0.5):
    '''Applies the pair crossover to a whole population:
    
    Args:
      population: list of individuals.
            
      crossover_rate: value between 0 and 1.
      
    Returns:
      population: list of individuals.
    '''  
    new_population = []
    
    if len(population) % 2 == 0:
        for parent1, parent2 in zip(population[ : : 2], population[1 : : 2]):
            individual1, individual2 = pair_crossover_cp(parent1, parent2, crossover_rate)
            new_population.append(individual1)
            new_population.append(individual2)
        rd.shuffle(new_population)
        return new_population
    
    else: 
        for parent1, parent2 in zip(population[ : -1 : 2], population[1 : : 2]):
            individual1, individual2 = pair_crossover_cp(parent1, parent2, crossover_rate)
            new_population.append(individual1)
            new_population.append(individual2)
            
        new_population += [population[-1]]
        rd.shuffle(new_population)
        return new_population

##################################################################################    
############################### SELECTION OPERATORS ##############################
##################################################################################
    
def tournament_selection_cp(population, fitness, num_individuals=3):
    """Applies selection by tournament.

    Args:
      population: list of individuals.
      
      fitness: list of fitness values from the individuals.
      
      num_individuals: number of solutions that will participate in the tournament.
            
    Returns:
      new_population: list of individuals.

    """
    new_population = []

    for _ in range(len(population)):
        tournament = rd.sample(population, num_individuals)

        tournament_fitness = []
        for individual in tournament:
            index = population.index(individual)
            tournament_fitness.append(fitness[index])

        best_fitness = max(tournament_fitness)
        index_best_fitness = tournament_fitness.index(best_fitness)
        selected = tournament[index_best_fitness]

        new_population.append(selected)

    return new_population

def roulette_selection_cp(population, fitness):
    """Applies selection by roullete.

    Args:
      population: list of individuals.
      
      fitness: list of fitness values from the individuals.
            
    Returns:
      new_population: list of individuals.

    """
    positive_fitness = [i + 2 for i in fitness]
    new_population = rd.choices(population, positive_fitness, k=len(population))
    return new_population