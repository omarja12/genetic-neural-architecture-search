# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 08:22:26 2022

@author: Omar Ja
"""

from EDA import *
from Individual import *

def generate_children(Parent1, Parent2, crossover_probability=CROSSOVER_PROBABILITY):
    """
    """
    Child1, Child2 = Parent1, Parent2
            
    # The place where to cut the network
    layers_min = min(len(Parent1.layer_layout), len(Parent2.layer_layout))
    layers_cut = np.random.choice(range(layers_min))
        
    if random.random() <= crossover_probability:
            
        # Apply cross over on the Loss function
        c1_loss, c2_loss = Parent2.loss, Parent1.loss
        # Apply cross over on Optimizer    
        c1_optimizer, c2_optimizer = Parent2.optimizer, Parent1.optimizer 
                
        # Apply cross over on Hidden Layers 
        c1_layer_layout = Parent1.layer_layout[:layers_cut] + Parent2.layer_layout[layers_cut:]
        c2_layer_layout = Parent2.layer_layout[:layers_cut] + Parent1.layer_layout[layers_cut:]
               
        Child1 = Individual(layer_layout=c1_layer_layout, optimizer=c1_optimizer, loss=c1_loss)
        Child2 = Individual(layer_layout=c2_layer_layout, optimizer=c2_optimizer, loss=c2_loss)
        
    return Child1, Child2
