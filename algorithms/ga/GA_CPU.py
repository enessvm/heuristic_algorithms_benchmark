import numpy as np

class GeneticAlgorithmCPU:
    def __init__(self, objective_func, pop_size, dim, max_iter, crossover_prob = 1, mutation_prob = 0.15, elitism_count = 2, minimize=True):
        """
        Parameters:
        - objective_func: Objective function to optimize.
        - pop_size: Number of individuals in the population.
        - dim: Number of dimensions.
        - max_iter: Maximum number of iterations.
        - crossover_prob: Probability of crossover.
        - mutation_prob: Probability of mutation.
        - elitism_count: Number of elites to keep.
        - minimize: Boolean to minimize (default) or maximize.
        """
        self.objective_func = objective_func
        self.pop_size = pop_size
        self.dim = dim
        self.max_iter = max_iter
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elitism_count = elitism_count
        self.minimize = minimize

        # Initialize population and other variables
        # - lb and ub: Lower and upper bounds for the search space.
        # - population: A randomly initialized array of individuals, where each individual is a vector of dimensionality `dim`.
        # - best_fitness: Tracks the best fitness value found during optimization.
        # - best_solution: Stores the individual corresponding to the best fitness.
        # - convergence_curve: A list that records the best fitness value at each iteration.
        self.lb = objective_func.lb  
        self.ub = objective_func.ub  
        self.population = np.random.uniform(self.lb, self.ub, (pop_size, dim))  
        self.best_fitness = float('inf') if minimize else float('-inf')  
        self.best_solution = None  
        self.convergence_curve = [] 

    def __evaluate_fitness(self):
        """Evaluate the fitness of the current population."""
        fitness = np.array([self.objective_func.evaluate_on_cpu(ind) for ind in self.population])
        return fitness
    
    def __tournament_selection(self, fitness, k=3):
        """Tournament selection: selects the best individual among k random individuals."""
        selected = []
        for _ in range(self.pop_size):
            candidates_idx = np.random.choice(len(self.population), k, replace=False)
            if self.minimize:
                best_idx = candidates_idx[np.argmin(fitness[candidates_idx])]
            else:
                best_idx = candidates_idx[np.argmax(fitness[candidates_idx])]
            selected.append(self.population[best_idx])
        return np.array(selected)
    
    def __crossover(self, parents):
        """One-point crossover for a pair of parents."""
        offspring = parents.copy()
        for i in range(0, len(parents), 2):
            if np.random.rand() < self.crossover_prob:
                point = np.random.randint(1, self.dim)
                offspring[i, point:], offspring[i + 1, point:] = (
                    parents[i + 1, point:],
                    parents[i, point:],
                )
        return offspring
    
    def __mutation(self, offspring):
        """Mutation: randomly changes a gene within the range [lb, ub]."""
        for individual in offspring:
            if np.random.rand() < self.mutation_prob:
                gene_idx = np.random.randint(self.dim)
                individual[gene_idx] = np.random.uniform(self.lb, self.ub)
        return offspring
    
    def run(self):
        """Run the genetic algorithm."""

        # Re-initialize variables for multiple runs
        self.population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        self.best_fitness = float('inf') if self.minimize else float('-inf')
        self.best_solution = None
        self.convergence_curve = []


        for generation in range(self.max_iter):
            # Evaluate fitness
            fitness = self.__evaluate_fitness()

            if self.minimize:
                elite_indices = np.argsort(fitness)[:self.elitism_count]
            else:
                elite_indices = np.argsort(fitness)[-self.elitism_count:]  
            elites = self.population[elite_indices]


            # Tournament selection
            selected_population = self.__tournament_selection(fitness)

            # Crossover
            offspring = self.__crossover(selected_population)

            # Mutation
            offspring = self.__mutation(offspring)

            # Replace old population with offspring
            self.population = offspring
            self.population[:self.elitism_count] = elites

            # Track the best solution
            current_best_idx = np.argmin(fitness) if self.minimize else np.argmax(fitness)
            current_best_fitness = fitness[current_best_idx]
            if (self.minimize and current_best_fitness < self.best_fitness) or \
               (not self.minimize and current_best_fitness > self.best_fitness):
                self.best_fitness = current_best_fitness
                self.best_solution = self.population[current_best_idx]


            self.convergence_curve.append(self.best_fitness)

            # Log progress every 10 generations
            if generation % 10 == 0:
                print(f"Generation {generation}: Best Fitness = {self.best_fitness:.6f}")
            

        return self.best_solution, self.best_fitness
