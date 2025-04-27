import random as rd
import math as mt

def cities_ts(number_cities, maximum_coordinate=100):
    '''Creates a certain number of cities in coordinate format.
    
    Args:
      number_cities: number of cities generated.
      
      maximum_coordinate: maximum integer that a coordinate can assume.
    
    Returns:
      cities: list of coordinates representing cities.
      '''
    cities = []
    
    while len(cities) < number_cities: 
        city = (rd.randint(0, maximum_coordinate), rd.randint(0, maximum_coordinate))
        if city not in cities:
            cities.append(city)
    return cities
    
def individual_ts(number_cities):
    '''Creates a sequence of cities with size equal to number_cities - 1. We ommit the starting city, number 0, and the 
    last city, which also would be city number 0.
    
    Args:
      number_cities: integer containing the number of cities.
      
    Returns:
      individual: list representing the city sequence.
      '''
    cities = [_ for _ in range(1, number_cities)]
    individual = rd.sample(cities, len(cities))
    return individual

def population_ts(population_size, number_cities):
    '''Creates a population of solutions.
    
    Args:
      population_size: number of individuals in the population.
      
      number_cities: integer containing the number of cities.
      
    Returns:
      population: list of individuals.
    '''
    population = []
    
    for _ in range(population_size):
        individual = individual_ts(number_cities)
        population.append(individual)
        
    return population
        
def individual_fitness_ts(individual, cities, penalty=1):
    '''Calculates the total distance in the path followed by a solution. Penalizes even cities first, because of the 
    restriction of the problem.
    
    Args:
      individual: sequence of cities.
      
      cities: list of city coordinates.
      
      penalty: integer representing punishment for even cities coming first.
      
    Returns:
      fitness: integer.
    '''
    number_cities = len(cities)
    x0, y0 = cities[0][0], cities[0][1]
    fitness = mt.sqrt((x0 - cities[individual[0]][0])**2 + (y0 - cities[individual[0]][1])**2)
    fitness += mt.sqrt((cities[individual[-1]][0] - x0)**2 + (cities[individual[-1]][1] - y0)**2)
    
    for i in range(number_cities - 2):
        x1 = cities[individual[i]][0]
        x2 = cities[individual[i+1]][0]
        y1 = cities[individual[i]][1]
        y2 = cities[individual[i+1]][1]
        fitness += mt.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
    if penalty == 0:
        return fitness
        
    if number_cities % 2 == 0:
        for i in range(int(number_cities / 2)):
            if individual[i] % 2 == 0:
                fitness += penalty
    else:
        for i in range(int(number_cities / 2)):
            if individual[i] % 2 == 0:
                fitness += penalty
    
    return fitness 
        
def population_fitness_ts(population, cities, penalty=1):
    '''Calculates the fitness for a population.
    
    Args:
      population: list of individuals.
      
      cities: list of city coordinates.
      
      penalty: integer representing punishment for even cities coming first.
      
    Returns:
      population_fitness: integer.
    '''
    population_fitness = []
    
    for individual in population: 
        fitness = individual_fitness_ts(individual, cities, penalty)
        population_fitness.append(fitness)
        
    return population_fitness

##################################################################################    
############################### MUTATION OPERATORS ###############################
##################################################################################
        
def switch_mutation_ts(individual, mutation_rate=0.05):
    '''Exchanges the position of two genes, with a certain mutation rate.
    
    Args:
      individual: list representing a solution.
      
      mutation_rate: value between 0 and 1.
      
    Returns:
      individual: list representing a solution.
    '''
    value = rd.random()
    individual_size = len(individual)
    
    if value < mutation_rate:
        index1, index2 = rd.randint(0, individual_size - 1), rd.randint(0, individual_size - 1)
        individual[index1], individual[index2] = individual[index2], individual[index1]
        
    return individual

def population_switch_mutation_ts(population, mutation_rate=0.05):
    '''Applies the gene switch mutation to a whole population:
    
    Args:
      population: list of individuals.
      
      mutation_rate: value between 0 and 1.
      
    Returns:
      population: list of individuals.
    
    '''    
    for individual in population:
        mutated_individual = switch_mutation_ts(individual, mutation_rate)    
    
    return population
        
##################################################################################    
############################### CROSSOVER OPERATORS ##############################
##################################################################################

def ordered_crossover_ts(parent1, parent2, crossover_rate=0.5):
    '''Applies the ordered crossover to two parents. It selects a partition of both parents, and creates two more lists 
    with them. After, it appends the rest of the elements that are not in the new individuals, following the order in which 
    they appear in the other parent.
    
    Args:
      parent1, parent2: possible solutions.
      
      crossover_rate: value between 0 and 1.
      
    Returns:
      individual1, individual2: possible solutions.
    '''
    size = len(parent1)
    start_index = rd.randint(0, size - 2)
    end_index = rd.randint(start_index + 2, size)

    individual1 = parent1[start_index : end_index]
    individual2 = parent2[start_index : end_index]
    
    for gene in parent2:
        if gene not in individual1:
            individual1.append(gene)
    
    for gene in parent1:
        if gene not in individual2:
            individual2.append(gene)
    
    return individual1, individual2
 
def population_ordered_crossover_ts(population, crossover_rate=0.5):
    '''Applies the ordered crossover to a whole population:
    
    Args:
      population: list of individuals.
            
      crossover_rate: value between 0 and 1.
      
    Returns:
      population: list of individuals.
    '''  
    new_population = []
    
    if len(population) % 2 == 0:
        for parent1, parent2 in zip(population[ : : 2], population[1 : : 2]):
            individual1, individual2 = ordered_crossover_ts(parent1, parent2, crossover_rate)
            new_population.append(individual1)
            new_population.append(individual2)
        rd.shuffle(new_population)
        return new_population
    
    else: 
        for parent1, parent2 in zip(population[ : -1 : 2], population[1 : : 2]):
            individual1, individual2 = ordered_crossover_ts(parent1, parent2, crossover_rate)
            new_population.append(individual1)
            new_population.append(individual2)
            
        new_population += [population[-1]]
        rd.shuffle(new_population)
        return new_population

##################################################################################    
############################### SELECTION OPERATORS ##############################
##################################################################################
    
def roulette_selection_ts(population, fitness):
    """Applies selection by roullete.

    Args:
      population: list of individuals.
      
      fitness: list of fitness values from the individuals.
            
    Returns:
      nem_population: list of individuals.

    """
    inverse_fitness = [(1/i) for i in fitness]
    new_population = rd.choices(population, inverse_fitness, k=len(population))
    return new_population



