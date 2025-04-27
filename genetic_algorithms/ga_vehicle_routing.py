import random as rd
import math as mt

def cities_vr(number_cities, maximum_coordinate=100):
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
    
def individual_vr(number_cities, number_vehicles):
    '''Creates a sequence of cities with size equal to number_cities - 1. We ommit the starting city, number 0, and the 
    last city, which also would be city number 0. Then, it separates the list into separate routes, one for each vehicle. 
    
    Args:
      number_cities: integer containing the number of cities.
      
      number_vehicles: integer number of vehicles in the problem.
      
    Returns:
      individual: list representing the city sequence, with each sublist being the route of one vehicle.
      '''
    individual = []
    cities = [_ for _ in range(1, number_cities)]
    total_individual = rd.sample(cities, len(cities)) 
    partitions = rd.sample([_ for _ in range(1, number_cities - 1)], number_vehicles - 1)
    partitions.sort()
    
    for i in range(len(partitions)):
        route = total_individual[0 : partitions[0]]
        if route not in individual:
            individual.append(route)
        else:
            route = total_individual[partitions[i - 1] : partitions[i]]
            individual.append(route)
    
    route = total_individual[partitions[-1] : ]
    individual.append(route)
            
    return individual

def population_vr(population_size, number_cities, number_vehicles):
    '''Creates a population of solutions.
    
    Args:
      population_size: number of individuals in the population.
      
      number_cities: integer containing the number of cities.
      
      number_vehicles: integer number of vehicles in the problem.
      
    Returns:
      individual: list representing the city sequence, with each sublist being the route of one vehicle.
      '''
    population = []
    
    for _ in range(population_size):
        individual = individual_vr(number_cities, number_vehicles)
        population.append(individual)
        
    return population
        
def individual_fitness_vr(individual, cities):
    '''Calculates the total distance in the path followed by a solution, including all routes. 
    
    Args:
      individual: sequence of routes.
      
      cities: list of city coordinates.
            
    Returns:
      fitness: integer.
    '''
    x0, y0 = cities[0][0], cities[0][1]
    fitness = 0
    for route in individual:
        number_cities = len(route)
        fitness += mt.sqrt((x0 - cities[route[0]][0])**2 + (y0 - cities[route[0]][1])**2)
        fitness += mt.sqrt((cities[route[-1]][0] - x0)**2 + (cities[route[-1]][1] - y0)**2)

        for i in range(number_cities - 1):
            x1 = cities[route[i]][0]
            x2 = cities[route[i+1]][0]
            y1 = cities[route[i]][1]
            y2 = cities[route[i+1]][1]
            fitness += mt.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    return fitness 
        
def population_fitness_vr(population, cities):
    '''Calculates the fitness for a population.
    
    Args:
      population: list of individuals.
      
      cities: list of city coordinates.
            
    Returns:
      population_fitness: integer.
    '''
    population_fitness = []
    
    for individual in population: 
        fitness = individual_fitness_vr(individual, cities)
        population_fitness.append(fitness)
        
    return population_fitness

##################################################################################    
############################### MUTATION OPERATORS ###############################
##################################################################################
        
def switch_mutation_vr(individual, individual_mutation_rate=0.05, route_mutation_rate=0.25):
    '''Exchanges the position of two cities in one ore more routes, with a certain mutation rate.
    
    Args:
      individual: list representing a solution.
      
      individual_mutation_rate: value between 0 and 1.
      
      route_mutation_rate: value between 0 and 1.
      
    Returns:
      individual: list representing a solution.
    '''
    routes = len(individual)
    value = rd.random()
    
    if value < individual_mutation_rate: 
        for route in individual:
            value = rd.random()
            if value < route_mutation_rate:
                route_size = len(route)
                index1, index2 = rd.randint(0, route_size - 1), rd.randint(0, route_size - 1)
                route[index1], route[index2] = route[index2], route[index1]

    return individual

def population_switch_mutation_vr(population, individual_mutation_rate=0.05, route_mutation_rate=0.25):
    '''Applies the gene switch mutation to a whole population:
    
    Args:
      population: list of individuals.
      
      individual_mutation_rate: value between 0 and 1.
      
      route_mutation_rate: value between 0 and 1.
      
    Returns:
      population: list of individuals.
    
    '''    
    for individual in population:
        mutated_individual = switch_mutation_vr(individual, individual_mutation_rate, route_mutation_rate)    
    
    return population

def partition_mutation_vr(individual, mutation_rate=0.05):
    '''Partitionates the solution in a different way. 
    
    Args:
      individual: list representing a solution.
      
      mutation_rate: value between 0 and 1.
      
    Returns:
      new_individual: list representing a solution.
    '''
    value = rd.random()
    if value < mutation_rate:
        number_vehicles = len(individual)
        total_individual = []
        
        for route in individual:
            total_individual += route
            
        number_cities = len(total_individual)
        new_individual = []
        partitions = rd.sample([_ for _ in range(1, number_cities - 2)], number_vehicles - 1)
        partitions.sort()

        for i in range(len(partitions)):
            route = total_individual[0 : partitions[0]]
            if route not in new_individual:
                new_individual.append(route)
            else:
                route = total_individual[partitions[i - 1] : partitions[i]]
                new_individual.append(route)

        route = total_individual[partitions[-1] : ]
        new_individual.append(route)

        return new_individual
            
def population_partition_mutation_vr(population, mutation_rate=0.05):
    '''Applies the partition mutation to a whole population:
    
    Args:
      population: list of individuals.
      
      mutation_rate: value between 0 and 1.
      
    Returns:
      population: list of individuals.
    
    '''    
    for individual in population:
        mutated_individual = partition_mutation_vr(individual, mutation_rate)    
    
    return population
        
##################################################################################    
############################### CROSSOVER OPERATORS ##############################
##################################################################################

def ordered_crossover_vr(parent1, parent2, crossover_rate=0.5):
    '''Applies the ordered crossover to two parents. It selects a partition of both parents, and creates two more lists 
    with them. After, it appends the rest of the elements that are not in the new individual, following the order in which 
    they appear in the other parent. To adequate to this problem, we concatenate the routes of each individual first, 
    apply the crossover, and then partition the new solutions in the same spots again.
    
    Args:
      parent1, parent2: possible solutions.
      
      crossover_rate: value between 0 and 1.
      
    Returns:
      individual1, individual2: possible solutions.
    '''
    value = rd.random()
    if value < crossover_rate:
        individual1, total_parent1, partitions1 = [], [], []
        individual2, total_parent2, partitions2 = [], [], []
        partition = 0

        for route in parent1:
            total_parent1 += route
            partition += len(route)
            partitions1.append(partition)

        partition = 0

        for route in parent2:
            total_parent2 += route
            partition += len(route)
            partitions2.append(partition)

        start_index = rd.randint(0, len(total_parent1) - 2)
        end_index = rd.randint(start_index + 2, len(total_parent1))

        total_individual1 = total_parent1[start_index : end_index]
        total_individual2 = total_parent2[start_index : end_index]

        for gene in total_parent2:
            if gene not in total_individual1:
                total_individual1.append(gene)

        for gene in total_parent1:
            if gene not in total_individual2:
                total_individual2.append(gene)

        for i in range(len(partitions1)):
            route = total_individual1[0 : partitions1[0]]
            if route not in individual1:
                individual1.append(route)
            else:
                route = total_individual1[partitions1[i - 1] : partitions1[i]]
                individual1.append(route)

        for i in range(len(partitions2)):
            route = total_individual2[0 : partitions2[0]]
            if route not in individual2:
                individual2.append(route)
            else:
                route = total_individual2[partitions2[i - 1] : partitions2[i]]
                individual2.append(route)

        return individual1, individual2
    
    else:
        return parent1, parent2

def population_ordered_crossover_vr(population, crossover_rate=0.5):
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
            individual1, individual2 = ordered_crossover_vr(parent1, parent2, crossover_rate)
            new_population.append(individual1)
            new_population.append(individual2)
        rd.shuffle(new_population)
        return new_population
    
    else: 
        for parent1, parent2 in zip(population[ : -1 : 2], population[1 : : 2]):
            individual1, individual2 = ordered_crossover_vr(parent1, parent2, crossover_rate)
            new_population.append(individual1)
            new_population.append(individual2)
            
        new_population += [population[-1]]
        rd.shuffle(new_population)
        return new_population

##################################################################################    
############################### SELECTION OPERATORS ##############################
##################################################################################
    
def roulette_selection_vr(population, fitness):
    """Applies selection by roullete.

    Args:
      population: list of individuals.
      
      fitness: list of fitness values from the individuals.
            
    Returns:
      new_population: list of individuals.

    """
    inverse_fitness = [(1/i) for i in fitness]
    new_population = rd.choices(population, inverse_fitness, k=len(population))
    return new_population

def tournament_selection_vr(population, fitness, num_individuals=3):
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

        best_fitness = min(tournament_fitness)
        index_best_fitness = tournament_fitness.index(best_fitness)
        selected = tournament[index_best_fitness]

        new_population.append(selected)

    return new_population
