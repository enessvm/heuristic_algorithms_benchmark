import numpy as np

class WhaleOptimizationCPU:
    def __init__(self, objective_func, pop_size, dim, max_iter, minimize=True):
        """
        Parameters:
        - objective_func: Objective function to optimize.
        - pop_size: Number of whales (solutions).
        - dim: Number of dimensions of the search space.
        - max_iter: Maximum number of iterations.
        - minimize: Boolean to minimize (default) or maximize.
        """
        self.objective_func = objective_func
        self.pop_size = pop_size
        self.dim = dim
        self.max_iter = max_iter
        self.minimize = minimize

        # Initialize positions and other variables:
        # - lb and ub: Lower and upper bounds for the search space.
        # - positions: Randomly initialized whale positions within the bounds, each position is a vector of length `dim`.
        # - best_fitness: Tracks the best fitness value found during optimization.
        # - best_solution: Stores the individual corresponding to the best fitness.
        # - convergence_curve: Records the best fitness value at each iteration.
        self.lb = objective_func.lb
        self.ub = objective_func.ub
        self.positions = self.__initialize_population()
        self.convergence_curve = np.zeros(max_iter)
        self.best_fitness = np.inf if minimize else -np.inf
        self.best_solution = np.zeros(dim)

    def __initialize_population(self):
        """Initialize population within bounds."""
        return np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))

    def __adjust_positions(self, position):
        """Ensure positions are within bounds."""
        return np.clip(position, self.lb, self.ub)

    def __calculate_fitness(self):
        """Calculate fitness for all whales and update the leader."""
        for i in range(self.pop_size):
            self.positions[i] = self.__adjust_positions(self.positions[i])
            fitness = self.objective_func.evaluate_on_cpu(self.positions[i])

            if (self.minimize and fitness < self.best_fitness) or (not self.minimize and fitness > self.best_fitness):
                self.best_fitness = fitness
                self.best_solution = self.positions[i].copy()

    def __update_positions(self, a):
        """Update whale positions based on WOA equations."""
        for i in range(self.pop_size):
            
            r1 = np.random.rand()
            r2 = np.random.rand()
            A = 2 * a * r1 - a
            C = 2 * r2
            b = 1
            l = np.random.uniform(-1, 1)
            p = np.random.rand()

            for j in range(self.dim):
                if p < 0.5:
                    if np.abs(A) < 1:
                        D_Leader = np.abs(C * self.best_solution[j] - self.positions[i][j])
                        self.positions[i][j] = self.best_solution[j] - A * D_Leader
                    else:
                        rand_idx = np.random.randint(0, self.pop_size)
                        random_pos = self.positions[rand_idx]
                        D_X_rand = np.abs(C * random_pos[j] - self.positions[i][j])
                        self.positions[i][j] = random_pos[j] - A * D_X_rand
                else:
                    distance2Leader = np.abs(self.best_solution[j] - self.positions[i][j])
                    self.positions[i][j] = (
                        distance2Leader * np.exp(b * l) * np.cos(2 * np.pi * l) + self.best_solution[j]
                    )

            self.positions[i] = self.__adjust_positions(self.positions[i])

    def run(self):
        """Run the Whale Optimization Algorithm."""

        # Re-initialize variables for multiple runs
        self.positions = self.__initialize_population()
        self.best_fitness = np.inf if self.minimize else -np.inf
        self.best_solution = np.zeros(self.dim)
        self.convergence_curve = np.zeros(self.max_iter)


        for iter in range(self.max_iter):
            self.__calculate_fitness()
            self.convergence_curve[iter] = self.best_fitness


            if iter % 10 == 0:
                print(f"Iteration {iter}: Best Fitness {self.best_fitness}")

            a = 2 - iter * (2 / self.max_iter)  # Linearly decreases from 2 to 0

            self.__update_positions(a)

        self.__calculate_fitness()
        return self.best_solution, self.best_fitness
