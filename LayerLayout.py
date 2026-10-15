# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 16:47:02 2022

@author: Omar Ja
"""

from EDA import *

class LayerLayout:

    """
    Define a Single Layer Layout
    """

    def __init__(self):
        
        self.layer_type = np.random.choice(HIDDEN_LAYER_TYPE)
    
        if self.layer_type == 'dense':
            self.neurons = np.random.choice(HIDDEN_LAYER_NEURONS)
            self.activation = np.random.choice(HIDDEN_LAYER_ACTIVATIONS)

        elif self.layer_type == 'dropout':
            self.rate = np.random.choice(HIDDEN_LAYER_RATE)
