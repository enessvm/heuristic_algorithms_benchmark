import numpy as np

class AntLionOptimizationCPU:
    def __init__(self, objective_func, pop_size, dim, max_iter, minimize=True):
        """
        Parameters:
        - objective_func: Objective function to optimize.
        - pop_size: Number of individuals in the population.
        - dim: Number of dimensions.
        - max_iter: Maximum number of iterations.
        - minimize: Boolean to minimize (default) or maximize.
        """
        self.objective_func = objective_func
        self.pop_size = pop_size
        self.dim = dim
        self.max_iter = max_iter
        self.minimize = minimize

        # Initialize population and other variables
        # - lb and ub: Lower and upper bounds for the search space.
        # - population: A randomly initialized array of individuals, where each individual is a vector of dimensionality `dim`.
        # - best_fitness: Tracks the best fitness value found during optimization.
        # - best_solution: Stores the individual corresponding to the best fitness.
        # - convergence_curve: A list that records the best fitness value at each iteration.
        self.lb = objective_func.lb  
        self.ub = objective_func.ub  
        self.population = self.__initialize_population()  
        self.convergence_curve = np.zeros(max_iter) 
        self.best_fitness = np.inf if minimize else -np.inf
        self.best_solution = np.zeros(dim)

    def __initialize_population(self):
        """Initialize population within bounds."""
        return np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))    

    def __random_walk_antlion(self, solution, current_iter):
        I = 1 

        if current_iter > self.max_iter / 10:
            I = 1 + 100 * (current_iter / self.max_iter)
        if current_iter > self.max_iter / 2:
            I = 1 + 1000 * (current_iter / self.max_iter)
        if current_iter > self.max_iter * (3 / 4):
            I = 1 + 10000 * (current_iter / self.max_iter)
        if current_iter > self.max_iter * 0.9:
            I = 1 + 100000 * (current_iter / self.max_iter)
        if current_iter > self.max_iter * 0.95:
            I = 1 + 1000000 * (current_iter / self.max_iter)

        # Decrease boundaries to converge towards antlion
        lb = np.full(self.dim, self.lb) / I 
        ub = np.full(self.dim, self.ub) / I

    
        # Move the interval of [lb ub] around the antlion [lb+anlion ub+antlion]
        lb = lb + solution if np.random.rand() < 0.5 else -lb + solution
        ub = ub + solution if np.random.rand() < 0.5 else -ub + solution

        # Normalize according to lb and ub vectors,
        X = np.array([np.cumsum(2 * (np.random.rand(self.pop_size) > 0.5) - 1) for _ in range(0, self.dim)])
        a = np.min(X, axis=1)
        b = np.max(X, axis=1)
        
        temp1 = np.reshape((ub - lb) / (b - a), (self.dim, 1))
        temp0 = X - np.reshape(a, (self.dim, 1))
        X_norm = temp0 * temp1 + np.reshape(lb, (self.dim, 1))
        return X_norm

    def __get_index_roulette_wheel(self, fitness):
        """Get index using roulette wheel selection."""
        if np.ptp(fitness) == 0:
            return np.random.randint(0, self.pop_size)

        adjusted_fitness = fitness if not self.minimize else np.max(fitness) - fitness
        
        prob = adjusted_fitness / np.sum(adjusted_fitness)

        return int(np.random.choice(range(self.pop_size), p=prob))

    def __adjust_population(self, population):
        """Ensure positions are within bounds."""
        return np.clip(population, self.lb, self.ub)

    def __calculate_fitness(self, population):
        """Calculate fitness for all ants and update the best solution."""
        return np.array([self.objective_func.evaluate_on_cpu(ind) for ind in population])

    def run(self):
        """Run the Ant Lion Optimizer."""
        
        # Re-initialize variables for multiple runs
        self.population = self.__initialize_population()
        self.best_fitness = np.inf if self.minimize else -np.inf
        self.best_solution = np.zeros(self.dim)
        self.convergence_curve = np.zeros(self.max_iter)

        for iter in range(self.max_iter):
            fitness = self.__calculate_fitness(self.population)


            new_population = np.zeros((self.pop_size, self.dim))
            for i in range(self.pop_size):

                roulette_index = self.__get_index_roulette_wheel(fitness)
                
                # RA is the random walk around the selected antlion by rolette wheel                
                RA = self.__random_walk_antlion(self.population[roulette_index], iter) 
                
                # RE is the random walk around the elite (the best antlion so far)                
                RE = self.__random_walk_antlion(self.best_solution, iter)
                
                new_pop = (RA[:, i] + RE[:, i]) / 2
                new_population[i] = self.__adjust_population(new_pop)

            # Update fitness
            new_fitness = self.__calculate_fitness(new_population)

            # Update antlions
            for i in range(self.pop_size):
                if (self.minimize and new_fitness[i] < fitness[i]) or (not self.minimize and new_fitness[i] > fitness[i]):
                    self.population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            # Update elite
            elite_idx = np.argmin(fitness) if self.minimize else np.argmax(fitness)
            self.best_solution = self.population[elite_idx]
            self.best_fitness = fitness[elite_idx]

            self.convergence_curve[iter] = self.best_fitness

            # Log progress every 10 iterations
            if iter % 10 == 0:
                print(f"Iteration {iter}: Best Fitness = {self.best_fitness:.6f}")

        return self.best_solution, self.best_fitness
                        