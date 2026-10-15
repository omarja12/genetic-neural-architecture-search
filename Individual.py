# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 16:44:43 2022

@author: Omar Ja
"""

from EDA import *
from LayerLayout import *


class Individual:
    """
    Individual Class
    """

    def __init__(self, **kwargs):
        """
        """
        self.layer_layout = kwargs["layer_layout"]
        self.optimizer = kwargs["optimizer"]
        self.loss = kwargs["loss"]
        self.fitness = None 
        self.model = None
        

    def __getitem__(self, ind):
        """Get the parameters that would be ``ind``th in iteration
        Parameters
        ----------
        ind : int
            The iteration index
        Returns
        -------
        params : dict of str to any
            Equal to list(self)[ind]
        """
        if len(self.layer_layout) > ind:
            return self.layer_layout[ind]
        else:
            raise IndexError("index out of range")
        
    def __repr__(self):
        return f"""Individual(Model Depth = {len(self.layer_layout)}, Optimizer={self.optimizer}, Loss={self.loss}, fitness={self.fitness})"""
    
    def generate_model(self, x_train, y_train, x_val, y_val, patience=PATIENCE, epochs=EPOCHS):
        
        # Define Neural Network Topology
        nn_model = Sequential()
        # Define Input Layer
        nn_model.add(InputLayer(input_shape=(x_train.shape[-1],)))
        # Add Hidden Layers
        for layer in self.layer_layout:

            if layer.layer_type == 'dense':
                nn_model.add(
                    Dense(
                        layer.neurons,
                        activation=layer.activation
                    )
                )
            elif layer.layer_type == 'dropout':
                nn_model.add(
                    Dropout(rate=layer.rate)
                )

        # Define Output Layer
        nn_model.add(Dense(1, activation='sigmoid'))

        # Compile Neural Network
        nn_model.compile(optimizer=self.optimizer,
                        loss= self.loss, 
                        metrics=f1)

        # Fit Model with Data
        nn_model.fit(
            x_train, 
            y_train,
            validation_data=(x_val, y_val),
            callbacks = EarlyStopping(monitor='val_f1', mode='max', patience=patience, verbose=0, restore_best_weights=True),
            class_weight = dict(enumerate(class_weight.compute_class_weight('balanced', 
                                classes=np.unique(y_train), y=y_train))),
            epochs=epochs,
            verbose=0)

        # Update Model into Individual
        self.model = nn_model
    
    def score(self, x, y, threshold=THRESHOLD):
        y_pred = (self.model.predict(x, verbose=0) > threshold).astype(int)
        return metrics.f1_score(y, y_pred, average='macro')

    def calculate_fitness(self, x_val, y_val, threshold=THRESHOLD):
        # Fitness is macro F1 on the validation set, never on the test set.
        self.fitness = self.score(x_val, y_val, threshold)            
        
    # In this case we are just getting the Euclidean distance:
    # Maybe we can return some other type of distance, but let see.     
    def distance(self, other):
        """
        """
        dist = np.abs(self.fitness - other.fitness)
        return dist

    # @property
    # def fitness(self):
    #     return self.fitness
    
    # @fitness.setter
    # def fitness(self, f):
    #     self.fitness = f
        
    # @property
    # def model(self):
    #     return self.model
    
    # @model.setter
    # def model(self, m):
    #     self.model = m
