import cupy as cp
import numpy as np

class GeneticAlgorithmGPU:
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
        self.crossover_prob = cp.float64(crossover_prob)
        self.mutation_prob = cp.float64(mutation_prob)
        self.elitism_count = elitism_count
        self.minimize = minimize

        # Initialize population on GPU
        self.lb = cp.float64(objective_func.lb)
        self.ub = cp.float64(objective_func.ub)
        self.population = self.__initialize_population()

        # Initialize best fitness and solution
        self.best_fitness = cp.inf if minimize else -cp.inf
        self.best_solution = cp.zeros(self.dim, dtype=cp.float64)
        self.convergence_curve = cp.zeros(max_iter)

        # Allocate memory for offspring, fitness values, and selected population
        self.offspring = cp.empty_like(self.population)
        self.fitness_values = cp.empty(self.pop_size, dtype=cp.float64)
        self.selected_population = cp.empty_like(self.population)

        # Set up kernel launch configurations
        self.threads_per_block = 256
        self.blocks_per_grid = (self.pop_size + self.threads_per_block - 1) // self.threads_per_block
        
        # Load kernels
        self.cuda_kernels = self.__get_kernels()
        self.module = cp.RawModule(code=self.cuda_kernels)
        self.selection_kernel = self.module.get_function('selection_kernel')
        self.crossover_kernel = self.module.get_function('crossover_kernel')
        self.mutation_kernel = self.module.get_function('mutation_kernel')
        self.fitness_kernel = self.objective_func.evaluate_on_gpu()
    
    def __get_kernels(self):
        """Define CUDA kernels for selection, crossover, and mutation."""
        return r'''
        #include <curand_kernel.h>
        #define M_PI 3.14159265359 

        extern "C" __global__
        void selection_kernel(const double* population, const double* fitness, double* selected, 
                              int pop_size, int k, int dim, bool minimize, unsigned int seed) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < pop_size) {
                curandState state;
                curand_init(seed + idx, 0, 0, &state);

                // Tournament selection
                int best_idx = curand(&state) % pop_size;
                double best_fitness = fitness[best_idx];

                for (int i = 1; i < k; i++) {
                    int candidate_idx = curand(&state) % pop_size;
                    if ((minimize && fitness[candidate_idx] < best_fitness) ||
                        (!minimize && fitness[candidate_idx] > best_fitness)) {
                        best_idx = candidate_idx;
                        best_fitness = fitness[candidate_idx];
                    }
                }

                // Copy selected individual
                for (int i = 0; i < dim; i++) {
                    selected[idx * dim + i] = population[best_idx * dim + i];
                }
            }
        }

        extern "C" __global__
        void crossover_kernel(const double* parents, double* offspring, int dim, int pop_size, double crossover_rate, unsigned int seed) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < pop_size / 2) {
                curandState state;
                curand_init(seed + idx, 0, 0, &state);

                int parent1_idx = 2 * idx;
                int parent2_idx = 2 * idx + 1;

                if (curand_uniform(&state) < crossover_rate) {
                    // Perform one-point crossover
                    int point = curand(&state) % dim;
                    for (int i = 0; i < dim; i++) {
                        if (i < point) {
                            offspring[parent1_idx * dim + i] = parents[parent1_idx * dim + i];
                            offspring[parent2_idx * dim + i] = parents[parent2_idx * dim + i];
                        } else {
                            offspring[parent1_idx * dim + i] = parents[parent2_idx * dim + i];
                            offspring[parent2_idx * dim + i] = parents[parent1_idx * dim + i];
                        }
                    }
                } else {
                    // Copy parents directly
                    for (int i = 0; i < dim; i++) {
                        offspring[parent1_idx * dim + i] = parents[parent1_idx * dim + i];
                        offspring[parent2_idx * dim + i] = parents[parent2_idx * dim + i];
                    }
                }
            }
        }

        extern "C" __global__
        void mutation_kernel(double* offspring, int dim, int pop_size, double mutation_rate, 
                             double lb, double ub, unsigned int seed) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < pop_size) {
                curandState state;
                curand_init(seed + idx, 0, 0, &state);

                // Perform mutation with the given mutation rate
                if (curand_uniform(&state) < mutation_rate) {
                    // Select a random gene to mutate
                    int gene_idx = idx * dim + (curand(&state) % dim);
                    
                    // Assign a new random value within [lb, ub] range
                    offspring[gene_idx] = lb + curand_uniform(&state) * (ub - lb);
                }
            }
        }
        '''
    
    def __initialize_population(self):
        """Initialize the population within the bounds."""
        return cp.random.uniform(self.lb, self.ub, (self.pop_size, self.dim), dtype=cp.float64)

    def _evaluate_fitness(self):
        """Evaluate the fitness of the current population."""
        self.fitness_kernel((self.blocks_per_grid,), (self.threads_per_block,),
                            (self.population, self.fitness_values, self.pop_size, self.dim))

    def __tournament_selection(self, seed):
        """Tournament selection: selects the best individual among k random individuals."""
        self.selection_kernel((self.blocks_per_grid,), (self.threads_per_block,),
                              (self.population, self.fitness_values, self.selected_population,
                               self.pop_size, 3, self.dim, self.minimize, seed))

    def __crossover(self, seed):        
        """One-point crossover for a pair of parents."""    
        self.crossover_kernel((self.blocks_per_grid,), (self.threads_per_block,),
                              (self.selected_population, self.offspring, self.dim, self.pop_size,
                               self.crossover_prob, seed))

    def __mutation(self, seed):
        """Mutation: randomly changes a gene within the range [lb, ub]."""
        self.mutation_kernel((self.blocks_per_grid,), (self.threads_per_block,),
                             (self.offspring, self.dim, self.pop_size, self.mutation_prob,
                              self.lb, self.ub, seed))

    def run(self):
        """Run the genetic algorithm."""

        # Re-initialize variables for multiple runs
        self.population = self.__initialize_population()  
        self.best_fitness = cp.inf if self.minimize else -cp.inf
        self.best_solution = cp.zeros(self.dim, dtype=cp.float64)
        self.convergence_curve = cp.zeros(self.max_iter)
        self.offspring = cp.empty_like(self.population)
        self.fitness_values = cp.empty(self.pop_size, dtype=cp.float64)
        self.selected_population = cp.empty_like(self.population)
        elites = cp.zeros((self.elitism_count, self.dim), dtype=cp.float64)

        for generation in range(self.max_iter):

            self._evaluate_fitness()

            # Preserve elites
            elite_indices = cp.argsort(self.fitness_values)
            if not self.minimize:
                elite_indices = elite_indices[::-1]  # Reverse for maximization
            elite_indices = elite_indices[:self.elitism_count]
            elites = self.population[elite_indices]

            # Selection
            seed = np.random.randint(1e9)
            self.__tournament_selection(seed)

            # Crossover
            seed = np.random.randint(1e9)
            self.__crossover(seed)  

            # Mutation
            seed = np.random.randint(1e9)
            self.__mutation(seed)   

            # Replace population with offspring
            self.population = cp.copy(self.offspring)
            self.population[:self.elitism_count] = elites

            # Update the best solution
            best_idx = cp.argmin(self.fitness_values) if self.minimize else cp.argmax(self.fitness_values)
            if (self.minimize and self.fitness_values[best_idx] < self.best_fitness) or \
                (not self.minimize and self.fitness_values[best_idx] > self.best_fitness):
                self.best_fitness = self.fitness_values[best_idx].copy()
                self.best_solution = self.population[best_idx].copy()
          
            self.convergence_curve[generation] = self.best_fitness

            if generation % 10 == 0:
                print(f"Generation {generation}: Best Fitness = {self.best_fitness:.6f}")

        return self.best_solution, self.best_fitness
