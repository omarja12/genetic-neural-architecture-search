# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 08:12:29 2022

@author: Omar Ja
"""

from EDA import *
from Individual import *
    
def mutate_individual(parent, mutation_probability=MUTATION_PROBABILITY):
    """
    """

    # Copy the layout so a mutation never changes the parent or a sibling.
    mutated_individual = Individual(layer_layout=deepcopy(parent.layer_layout), optimizer=parent.optimizer, loss=parent.loss)
    if random.random() <= mutation_probability:
            
        # Apply Mutation on Optimizer
        mutated_individual.optimizer = np.random.choice(MODEL_OPTIMIZER)
        # Apply mutation on the Loss function
        mutated_individual.loss = np.random.choice(MODEL_LOSS)        
        # Apply Mutation on Hidden Layer Size
        new_hl_size = np.random.choice(HIDDEN_LAYER_COUNT)

        # Check if Need to Expand or Reduce Layer Count
        if new_hl_size > len(mutated_individual.layer_layout):

            # Increase Layer Count
            while len(mutated_individual.layer_layout) < new_hl_size:
                mutated_individual.layer_layout.append(LayerLayout())

        elif new_hl_size < len(mutated_individual.layer_layout):
            # Reduce Layers Count
            mutated_individual.layer_layout = mutated_individual.layer_layout[0: new_hl_size]
            
    return mutated_individual