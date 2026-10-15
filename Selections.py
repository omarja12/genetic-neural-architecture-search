# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 15:17:33 2022

@author: Omar Ja
"""

import random
import numpy as np
from random import sample
from operator import attrgetter

def tournament_selection(population, optim, size = 3):
    """

    """            
    # Select individuals based on tournament size
    tournament = sample(population, min(size, len(population)))
    # Check if the problem is a maximization or minimization
    if optim == "max":
        return max(tournament, key = attrgetter("fitness"))
    elif optim == "min":
        return min(tournament, key = attrgetter("fitness"))
    else:
        raise Exception("No optimization specified") 
    
def ranking_selection(population):
    """
    """
    # Check if the problem is max or min 
    if population.optim == "max":        
        sorted_population = sorted(population.individuals, key=attrgetter("fitness"))
    elif population.optim == "min":        
        sorted_population = sorted(population.individuals, key=attrgetter("fitness"), reverse = True)
    else:
        raise Exception("No optimization type specified")
    #
    total = sum(range(population.size+1))
    position = 0
    spin = random.random()
    #    
    for count, individual in enumerate(sorted_population):
        position+=(count+1)/total
        if spin < position:
            return individual


def fps_2(population):
    """Fitness proportionate selection implementation.
            Args:
                population (Population): The population we want to select from.
            Returns:
                Individual: selected individual.
    """ 
    # Sum total fitnesses
    total_fitness_max = sum([individual.fitness for individual in population.individuals])
    total_fitness_min = sum([1.0/individual.fitness for individual in population.individuals])    
            
    # Get a 'position' on the wheel
    position = 0
    spin = random.random()
    # 
    if population.optim == "max":
        # Find individual in the position of the spin
        for individual in population.individuals:  
            position += individual.fitness
            if spin <= position/total_fitness_max:
                return individual
    if population.optim == "min":
        # Find individual in the position of the spin
        for individual in population.individuals: 
            position += (1.0/individual.fitness)
            if spin <= position/total_fitness_min:
                return individual
    else:
        raise Exception("No optimization type specified") 