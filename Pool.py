# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 16:49:20 2022

@author: Omar Ja
"""

from Crossover import *
from EDA import *
from Individual import *
from LayerLayout import *
from Mutation import *
from Selections import *


class Pool:
    
    def __init__(self, **kwargs):
        """
        """
        self.size = kwargs["size"]
        self.optim = kwargs["optim"]
        self.x_train = kwargs["x_train"]
        self.y_train = kwargs["y_train"]
        self.x_val = kwargs["x_val"]
        self.y_val = kwargs["y_val"]
        
        self.individuals = []
        self.best_fitness_over_generations = []
        self.mean_fitness_over_generations = []
        self.sd_fitness_over_generations = []
        self.current_generation = 1
        self.best_fitness = None
        self.mean_fitness = None
        self.sd_fitness = None
        self.timestamp = int(time.time())
        self.dataframe = pd.DataFrame(columns=["Current Generation", "Best Individual",
                                               "Mean Fitness", "Sd Fitness", "Best Fitness"])
        
        while len(self.individuals) < self.size:
            # Choose Hidden Layer Count
            hidden_layer_count = np.random.choice(HIDDEN_LAYER_COUNT)
            hidden_layer_layout = []

            # Define Layer Structure
            for _ in range(hidden_layer_count):
                hidden_layer_layout.append(LayerLayout())
                
            individual = Individual(
                layer_layout=hidden_layer_layout,
                optimizer=np.random.choice(MODEL_OPTIMIZER),
                loss=np.random.choice(MODEL_LOSS)
                )

            self.individuals.append(individual)
   
    def __repr__(self):
        """
        """
        representation=f"""Population(size={len(self.individuals)}, Optimization_Type={self.optim}, Average_Fitness={self.mean_fitness}, Sd_Fitness={self.sd_fitness}, Best_Fitness={self.best_fitness})"""
        return representation    
    
    def __getitem__(self, ind):
        """
        If the index is within the range of the list, return the individual at that index. Otherwise, raise
        an error
        
        :param ind: the index of the individual you want to get
        :return: The individual at the index ind.
        """
        if len(self.individuals) > ind:
            return self.individuals[ind]
        else:
            raise IndexError("index out of range")

    def __len__(self):
        """
        The function returns the length of the list of individuals.
        :return: The length of the list of individuals.
        """
        return len(self.individuals)
    
    
    def fitnesses_statistics(self):
        """
        It calculates the mean, standard deviation, and best fitness of the population
        :return: The best fitness, mean fitness, and standard deviation of fitness.
        """
        """
        """
        mean_fitness = np.mean([individual.fitness for individual in self.individuals])
        sd_fitness = np.std([individual.fitness for individual in self.individuals])
        if self.optim == "max":
            best_fitness = max(self.individuals, key=attrgetter("fitness")).fitness 
        elif self.optim == "min":
            best_fitness = min(self.individuals, key=attrgetter("fitness")).fitness
        else:
            raise Exception("No optimization type specified")
                
        self.best_fitness = best_fitness
        self.mean_fitness = mean_fitness
        self.sd_fitness = sd_fitness
        return best_fitness,mean_fitness,sd_fitness            
        
    def select_elits(self, elit_size=ELITE_SIZE):
        """
        """
        if self.optim == "max":
            self.individuals.sort(key=attrgetter("fitness"), reverse=True)
        elif self.optim == "min":
            self.individuals.sort(key = attrgetter("fitness"))
        else:
            raise Exception("No optimization type specified")
        
        return self.individuals[0:elit_size], self.individuals[elit_size:] 
       
    def create_models(self, x_train, y_train):
        """
        """
        # Train all Models
        for individual in self.individuals:
            individual.generate_model(self.x_train, self.y_train, self.x_val, self.y_val)
                 
    def evaluate(self, population):
        """
        """
        for individual in population:
            
            # individual.model is assigned here
            individual.generate_model(self.x_train, self.y_train, self.x_val, self.y_val)
            
            # individual.fitness is assigned here, on the validation set.
            individual.calculate_fitness(self.x_val, self.y_val)
            
        # Sort individuals by fitness
        if self.optim == "max":
            population.sort(key=attrgetter("fitness"), reverse=True)
    
        elif self.optim == "min":
            population.sort(key=attrgetter("fitness"))
        else:
            raise Exception("No optimization type specified")
       
        return population
            
    def fitness_sharing(self, epsilon=10**-5):
        """Fitness sharing (fitness scaling). Not finished: use fitness_sharing=False.
        """
        raise NotImplementedError("fitness sharing is not implemented; run evolve with fitness_sharing=False")
        distance_matrix = [[None] for _ in range(self.size-1)]
        sharing_matrix = [[None] for _ in range(self.size-1)]
        sharing_array = [None]*(self.size-1)
        # should I define a martix or just an array of arrays.
        for i, indiv1 in enumerate(self.individuals):
            for j, indiv2 in enumerate(self.individuals):
                while i!=j:
                    d_i_j = indiv1.distance(indiv2)
                    distance_matrix[i].append(d_i_j)
            # for now i decide to divide only by the sum of the distances, maybe later i can try something like 
            # min mac encoding
            sharing_matrix[i] = [1]*(self.size-1) - distance_matrix[i]/sum(distance_matrix[i])/(self.size-1)
            sharing_array[i] = sum(sharing_matrix[i])+ epsilon # To avoid dividing by 0.
            indiv1.fitness /= sharing_array[i]   
        
    def evolve(self, elitism, fitness_sharing, generations=GENERATIONS):
        """
        > Evolve the population for a given number of generations, using elitism and fitness sharing
        
        :param elitism: the number of individuals to keep from one generation to the next
        :param fitness_sharing: If True, the fitness of each individual is divided by the number of
        individuals in its species
        :param generations: The number of generations to evolve the population for
        """
                  
        for gen in range(generations):
            new_individuals = []
            new_leftout_individuals = []
            leftout_individuals = self.individuals        
            if self.current_generation == 1:
                # Generate and evaluate the models.
                self.individuals = self.evaluate(self.individuals)
            # self.log(elitism)
            self.fitnesses_statistics()
            # Appending to list to be plot later
            self.best_fitness_over_generations.append(self.best_fitness)
            self.mean_fitness_over_generations.append(self.mean_fitness)
            self.sd_fitness_over_generations.append(self.sd_fitness) 
             
            # This is not working for now 
            if fitness_sharing == True:
                self.fitness_sharing()
                
            # Check if elitism is set to true
            if elitism == True:
                elit_individuals, leftout_individuals = self.select_elits()
                new_individuals.extend(elit_individuals)
            
            for idx, indiv in enumerate(leftout_individuals):
            # while len(new_individuals) < self.size:
                # Selecting two individuals
                # select is a function here i use tournament selection but this should be possible with any type of selection
                parent1, parent2 = tournament_selection(leftout_individuals, self.optim), tournament_selection(leftout_individuals, self.optim)
                
                # Crossover
                offspring1, offspring2 = generate_children(parent1, parent2)
                
                # Mutation                
                offspring1 = mutate_individual(offspring1)  
                offspring2 = mutate_individual(offspring2)

                new_leftout_individuals.append(offspring1)                        
                if len(new_leftout_individuals) < len(leftout_individuals):
                    new_leftout_individuals.append(offspring2)

                                                  
            
            new_leftout_individuals = self.evaluate(new_leftout_individuals)
            new_individuals.extend(new_leftout_individuals)
            self.individuals = new_individuals
            self.current_generation +=1
            self.fitnesses_statistics()
            # I can only run this after using evaluate.
            # Do I really need this as the evaluate fct already sort the population
            # if self.optim == "max":
            #     best_individual = max(self.individuals, key=lambda x: x.fitness)
            # if self.optim == "min":
            #     best_individual = min(self.individuals, key=lambda x: x.fitness)
            # else:
            #     raise Exception("No optimization type specified")
            
            
            
            # Adding a breaking condition so we can break out of the fct if the condition is verified.
            # I can add some other condition with lags to break out from the fct as well
            if self.best_fitness > 0.9:
                break
            
        # Record the final generation too.
        self.best_fitness_over_generations.append(self.best_fitness)
        self.mean_fitness_over_generations.append(self.mean_fitness)
        self.sd_fitness_over_generations.append(self.sd_fitness)
        return(self.best_fitness, self.best_fitness_over_generations,
               self.mean_fitness_over_generations, self.sd_fitness_over_generations)
        
    def log(self, elitism):
        if self.optim == "max":
            if elitism == False:
                best_model = max(self, key=attrgetter("fitness"))
                data = [self.current_generation, best_model, best_model.fitness]
                self.dataframe.loc[self.current_generation-1] = data
                # I need to find a name for this file
                # I THINK SHOULD NOT KEEP SAVIGN THE DATAFRAME HERE
                self.dataframe.to_csv(f'{self.timestamp}.csv', mode='w', index=False, header=True) 
            
            if elitism == True:
                best_model = max(self, key=attrgetter("fitness"))
                data = [self.current_generation, best_model, best_model.fitness]
                self.dataframe.loc[self.current_generation-1] = data
                # Again I need to find a good name for this file
                self.dataframe.to_csv(f'Elitism{self.timestamp}.csv', mode='w', index=False, header=True) 

        elif self.optim == "min":
            if elitism == False:
                best_model = min(self, key=attrgetter("fitness"))
                data = [self.current_generation, best_model, best_model.fitness]
                self.dataframe.loc[self.current_generation-1] = data
                # I need to find a name for this file
                self.dataframe.to_csv(f'{self.timestamp}.csv', mode='w', index=False, header=True) 
            
            if elitism == True:
                best_model = min(self, key=attrgetter("fitness"))
                data = [self.current_generation, best_model, best_model.fitness]
                self.dataframe.loc[self.current_generation-1] = data
                # Again I need to find a good name for this file
                self.dataframe.to_csv(f'Elitism{self.timestamp}.csv', mode='w', index=False, header=True)  
    
    
    # @property
    # def best_fitness(self):
    #     return self.best_fitness
    
    # @best_fitness.setter
    # def best_fitness(self, bf):
    #     self.best_fitness = bf
        
    # @property
    # def mean_fitness(self):
    #     return self.mean_fitness

    # @mean_fitness.setter
    # def mean_fitness(self, mf):
    #     self.mean_fitness = mf
        
    # @property
    # def sd_fitness(self):
    #     return self.sd_fitness
    
    # @sd_fitness.setter
    # def sd_fitness(self, sf):
    #     self.sd_fitness = sf
