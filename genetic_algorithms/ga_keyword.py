import random as rd
import string 

CHARACTERS = string.ascii_letters + string.digits

def gene_kw():
    '''Returns a possible character in a keyword, including upper and lowercase letters, and digits.
    
    Returns:
      char: list containing a uppercase or lowercase letter, or a digit.
      '''
    char = rd.choice(CHARACTERS)
    return char
    
def individual_kw(individual_size):
    '''Creates a keyword with size ranging from size[0] to size[-1].
    
    Args:
      individual_size: list containing the lower and upper limits of the keyword size.
      
    Returns:
      individual: list representing the keyword.
      '''
    size_kw = rd.choice([i for i in range(individual_size[0], individual_size[-1] + 1)])
    individual = []
    
    for _ in range(size_kw):
        gene = gene_kw()
        individual.append(gene)
        
    return individual

def population_kw(population_size, individual_size):
    '''Creates a population of keywords.
    
    Args:
      population_size: number of individuals in the population.
      
      individual_size: range of sizes of the keyword.
      
    Returns:
      population: list of individuals.
    '''
    population = []
    
    for _ in range(population_size):
        individual = individual_kw(individual_size)
        population.append(individual)
        
    return population
        
def individual_fitness_kw(individual, keyword):
    '''Computes the difference between two possible keywords, according to its characters and sizes. For each corresponding     character, it outputs 1, if they are different, and 0 otherwise. The overall fitness is the sum of all different 
    characters and the absolute value of the size difference between the individual and the keyword.
    
    
    Args:
      individual: candidate to being the keyword.
      
      keyword: actual keyword.
      
    Returns:
      fitness: integer 
    '''
    individual_size = len(individual)
    keyword_size = len(keyword)
    size_difference = abs(individual_size - keyword_size)
    
    if individual_size <= keyword_size:
        fitness = 0
        for i in range(individual_size):
            if individual[i] != keyword[i]:
                fitness += 1
                
        fitness += size_difference
        return fitness 
    
    else:
        fitness = 0
        for i in range(keyword_size):
            if individual[i] != keyword[i]:
                fitness += 1
                
        fitness += size_difference
        return fitness 
            
def population_fitness_kw(population, keyword):
    '''Calculates the fitness for a population.
    
    Args:
      population: list of individuals.
      
      keyword: actual keyword.
      
    Returns:
      population_fitness: list with all fitness values from the individuals in the population.
      
    '''
    population_fitness = []
    
    for individual in population: 
        fitness = individual_fitness_kw(individual, keyword)
        population_fitness.append(fitness)
        
    return population_fitness

##################################################################################    
############################### MUTATION OPERATORS ###############################
##################################################################################
        
def gene_mutation_kw(individual, mutation_rate=0.05):
    '''Mutates one gene from a individual, with a certain mutation rate.
    
    Args:
      individual: list representing a solution.
      
      mutation_rate: value between 0 and 1.
      
    Returns:
      individual: list representing a solution.
    '''
    value = rd.random()
    individual_size = len(individual)
    
    if value < mutation_rate:
        index = rd.randint(0, individual_size - 1)
        individual[index] = gene_kw()
        
    return individual

def population_gene_mutation_kw(population, mutation_rate=0.05):
    '''Applies the gene mutation to a whole population:
    
    Args:
      population: list of individuals.
      
      mutation_rate: value between 0 and 1.
      
    Returns:
      population: list of individuals.
    
    '''    
    for individual in population:
        mutated_individual = gene_mutation_kw(individual, mutation_rate)    
    
    return population
        
def size_mutation_kw(individual, size, mutation_rate=0.05):
    '''Mutates the size of an individual, with a certain mutation rate. When mutated, the individual gets additional genes, 
    or remover genes.
    
    Args:
      individual: list representing a solution.
      
      size: list containing minimum and maximum length of an individual. 
      
      mutation_rate: value between 0 and 1.
      
    Returns:
      individual: list representing a solution.
    '''
    value = rd.random()
    individual_size = len(individual)
    
    if value < mutation_rate:
        new_size = rd.randint(1, size[-1])
        size_difference = new_size - individual_size
        
        if size_difference > 0:
            for _ in range(size_difference):
                individual += gene_kw()
            return individual
        
        if size_difference < 0:
            for _ in range(1, abs(size_difference) + 1):
                individual.pop(-1)   
            return individual
        
def population_size_mutation_kw(population, size, mutation_rate=0.05):
    '''Applies the size mutation to a whole population:
    
    Args:
      population: list of individuals.
      
      size: list containing minimum and maximum length of an individual.
      
      mutation_rate: value between 0 and 1.
      
    Returns:
      population: list of individuals.
    
    '''    
    for individual in population:
        mutated_individual = size_mutation_kw(individual, size, mutation_rate)
        
    return population
        
##################################################################################    
############################### CROSSOVER OPERATORS ##############################
##################################################################################

def size_adaptative_crossover_kw(parent1, parent2, crossover_rate=0.5):
    '''Crossover operator that accounts different size individuals.
    
    Args:
      parent1, parent2: individuals representing solutions.
      
      crossover_rate: value between 0 and 1.
      
    Returns:
      individual1, individual2: individuals representing solutions.
    '''
    parent1_size = len(parent1)
    parent2_size = len(parent2)
    size_difference = parent1_size - parent2_size
    value = rd.random()
    
    if value < crossover_rate:
    
        if size_difference > 0:
            individual1 = parent2 + parent1[parent2_size :] 
            individual2 = parent1[0 : parent2_size]
            return individual1, individual2

        elif size_difference < 0:
            individual1 = parent1 + parent2[parent1_size :] 
            individual2 = parent2[0 : parent1_size]
            return individual1, individual2

        else:
            return parent1, parent2
    else: 
        return parent1, parent2
        
def population_size_adaptative_crossover_kw(population, crossover_rate=0.5):
    '''Applies the size adaptative crossover to a whole population:
    
    Args:
      population: list of individuals.
            
      crossover_rate: value between 0 and 1.
      
    Returns:
      population: list of individuals.
    '''  
    new_population = []
    
    if len(population) % 2 == 0:
        for parent1, parent2 in zip(population[ : : 2], population[1 : : 2]):
            individual1, individual2 = size_adaptative_crossover_kw(parent1, parent2, crossover_rate)
            new_population.append(individual1)
            new_population.append(individual2)
        rd.shuffle(new_population)
        return new_population
    
    else: 
        for parent1, parent2 in zip(population[ : -1 : 2], population[1 : : 2]):
            individual1, individual2 = size_adaptative_crossover_kw(parent1, parent2, crossover_rate)
            new_population.append(individual1)
            new_population.append(individual2)
            
        new_population += [population[-1]]
        rd.shuffle(new_population)
        return new_population

##################################################################################    
############################### SELECTION OPERATORS ##############################
##################################################################################
    
def roulette_selection_kw(population, fitness):
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



