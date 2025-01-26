import numpy as np
import cupy as cp

class AntLionOptimizationGPU:
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
        self.pop_size = cp.int32(pop_size)
        self.dim = cp.int32(dim)
        self.max_iter = max_iter
        self.minimize = minimize

        # Initialize population and other variables
        # - lb and ub: Lower and upper bounds for the search space.
        # - population: A randomly initialized array of individuals, where each individual is a vector of dimensionality `dim`.
        # - best_fitness: Tracks the best fitness value found during optimization.
        # - best_solution: Stores the individual corresponding to the best fitness.
        # - convergence_curve: A list that records the best fitness value at each iteration.
        self.lb = cp.float64(objective_func.lb) 
        self.ub = cp.float64(objective_func.ub)  
        self.population = self.__initialize_population()  
        self.convergence_curve = cp.zeros(max_iter) 
        self.best_fitness = cp.inf if minimize else -cp.inf
        self.best_solution = cp.zeros(dim, dtype=cp.float64)

        # CUDA kernel setup
        self.cuda_kernels = self.__get_kernels()
        self.module = cp.RawModule(code=self.cuda_kernels)
        self.calculate_bounds_kernel = self.module.get_function("calculate_bounds")
        self.random_walk_kernel = self.module.get_function("random_walk")
        self.update_population_kernel = self.module.get_function("update_population")
        self.replace_with_better_solution_kernel = self.module.get_function("replace_with_better_solution")

        self.fitness_kernel = self.objective_func.evaluate_on_gpu()

        # CUDA configuration
        self.threads_per_block = 256
        self.blocks_per_grid = (pop_size + self.threads_per_block - 1) // self.threads_per_block

    def __get_kernels(self):
        """Return the CUDA kernel code."""
        return r'''
        #include <curand_kernel.h>

        extern "C" __global__
        void calculate_bounds(const double* solutions, double I, 
                              const double* lb_array, const double* ub_array,
                              double* lb_out, double* ub_out, 
                              int dim, int pop_size, unsigned int seed) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= pop_size) return;

            curandState state;
            curand_init(seed, idx, 0, &state);

            int base = idx * dim;
            bool use_positive_lb = curand_uniform(&state) > 0.5;
            bool use_positive_ub = curand_uniform(&state) > 0.5;

            for (int i = 0; i < dim; ++i) {
                double solution = solutions[base + i];
                double lb = lb_array[i] / I;
                double ub = ub_array[i] / I;

                if (use_positive_lb) {
                    lb_out[base + i] = lb + solution;
                }
                else {
                    lb_out[base + i] = -lb + solution;
                }

                if (use_positive_ub) {
                    ub_out[base + i] = ub + solution;
                }
                else {
                    ub_out[base + i] = -ub + solution;
                }
            }
        }

        extern "C" __global__
        void random_walk(double* random_walks, const double* lb, const double* ub, 
                         int dim, int pop_size, unsigned int seed) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= pop_size) return;

            curandState state;
            curand_init(seed, idx, 0, &state);

            int base = idx * dim;
            for (int i = 0; i < dim; ++i) {
                double step = 0.0;
                double min_val = 0.0, max_val = 0.0;

                for (int j = 0; j < pop_size; ++j) {
                    step += (curand_uniform(&state) > 0.5 ? 1.0 : -1.0);
                    min_val = fmin(step, min_val);
                    max_val = fmax(step, max_val);
                }

                double scale = (ub[i] - lb[i]) / (max_val - min_val);
                random_walks[base + i] = (step - min_val) * scale + lb[i];
            }
        }

        extern "C" __global__
        void update_population(const double* RA, const double* RE, double* new_population, 
                               const double lb, const double ub, int dim, int pop_size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= pop_size) return;

            int base = idx * dim;
            for (int i = 0; i < dim; ++i) {
                double new_val = (RA[base + i] + RE[base + i]) / 2.0;
                new_val = fmax(fmin(new_val, ub), lb);

                new_population[base + i] = new_val;
            }
        }

        extern "C" __global__
        void replace_with_better_solution(double* population, double* fitness, 
                                        const double* new_population, const double* new_fitness,
                                        int pop_size, int dim, bool minimize) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= pop_size) return;

            if ((minimize && new_fitness[idx] < fitness[idx]) || 
                (!minimize && new_fitness[idx] > fitness[idx])) {
                // Update population and fitness if new fitness is better
                int base = idx * dim;

                for (int i = 0; i < dim; ++i) {
                    population[base + i] = new_population[base + i];
                }
                fitness[idx] = new_fitness[idx];
            }
        }
        '''

    def __initialize_population(self):
        """Initialize population within bounds."""
        return cp.random.uniform(self.lb, self.ub, size=(self.pop_size, self.dim), dtype=cp.float64)

    def __get_index_roulette_wheel(self, fitness):
        """Simulate a roulette wheel selection."""
        fitness = fitness.get()
        adjusted_fitness = (np.max(fitness) - fitness) if self.minimize else fitness
        adjusted_fitness = np.maximum(0, adjusted_fitness)

        if np.sum(adjusted_fitness) == 0:
            return np.random.randint(0, len(fitness), size=self.pop_size)

        probabilities = adjusted_fitness / np.sum(adjusted_fitness)
        cumulative_probabilities = np.cumsum(probabilities)
        rand = np.random.rand(self.pop_size)

        return np.searchsorted(cumulative_probabilities, rand)
    
    def __calculate_random_walks(self, solutions, I, lb_array, ub_array):
        """Calculate random walks for given solutions."""
        lb_out = cp.zeros_like(solutions)
        ub_out = cp.zeros_like(solutions)

        self.calculate_bounds_kernel(
            (self.blocks_per_grid,), (self.threads_per_block,),
            (solutions, I, lb_array, ub_array, lb_out, ub_out, self.dim, self.pop_size, np.random.randint(1e6))
        )

        random_walks = cp.zeros_like(solutions)
        self.random_walk_kernel(
            (self.blocks_per_grid,), (self.threads_per_block,),
            (random_walks, lb_out, ub_out, self.dim, self.pop_size, np.random.randint(1e6))
        )
        return random_walks

    def __calculate_fitness(self, population):
        """Calculate fitness for all individuals in the population."""
        fitness = cp.zeros(self.pop_size, dtype=cp.float64)

        self.fitness_kernel(
            (self.blocks_per_grid,), (self.threads_per_block,),
            (
                population,
                fitness,
                self.pop_size,
                self.dim
            )
        )

        return fitness
    
    def __calculate_I(self, current_iter):
        """Calculate the convergence factor I based on the iteration number."""
        I = 1.0
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
        return I


    def __update_population(self, fitness, iter):
        """Update the population using the antlion random walk strategy."""
        # Calculate convergence factor I
        I = self.__calculate_I(iter)

        # Prepare data
        lb_array = cp.full(self.dim, self.lb)
        ub_array = cp.full(self.dim, self.ub)
        selected_indices = cp.array(self.__get_index_roulette_wheel(fitness), dtype=cp.int32)
        selected_solutions = cp.array(self.population[selected_indices], dtype=cp.float64)

        # Calculate random walks
        RA = self.__calculate_random_walks(selected_solutions, I, lb_array, ub_array)
        RE = self.__calculate_random_walks(cp.tile(self.best_solution, (self.pop_size, 1)), 
                                           I, lb_array, ub_array)
        
        # Allocate memory for new population
        new_population = cp.zeros_like(self.population)

        # Launch kernel to update population
        self.update_population_kernel(
            (self.blocks_per_grid,), (self.threads_per_block,),
            (RA, RE, new_population, self.lb, self.ub, self.dim, self.pop_size)
        )

        # Calculate fitness of new population
        new_fitness = self.__calculate_fitness(new_population)
        return new_population, new_fitness

    def run(self):
        """Run the Ant Lion Optimizer."""
        self.population = self.__initialize_population()
        self.best_fitness = cp.inf if self.minimize else -cp.inf
        self.best_solution = cp.zeros(self.dim, dtype=cp.float64)
        self.convergence_curve = cp.zeros(self.max_iter)

        for iter in range(self.max_iter):
            fitness = self.__calculate_fitness(self.population)

            new_population, new_fitness = self.__update_population(fitness, iter)


            # Update the population and fitness values
            self.replace_with_better_solution_kernel(
                (self.blocks_per_grid,), (self.threads_per_block,),
                (self.population, fitness, new_population, new_fitness, self.pop_size, self.dim, self.minimize)
            )

            elite_idx = cp.argmin(fitness) if self.minimize else cp.argmax(fitness)
            self.best_solution = self.population[elite_idx]
            self.best_fitness = fitness[elite_idx]

            self.convergence_curve[iter] = self.best_fitness

            if iter % 10 == 0:
                print(f"Iteration {iter}: Best Fitness = {self.best_fitness:.6f}")

        return self.best_solution, self.best_fitness
                        