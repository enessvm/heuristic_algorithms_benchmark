import cupy as cp


class WhaleOptimizationGPU:
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
        self.pop_size = cp.int32(pop_size)
        self.dim = cp.int32(dim)
        self.max_iter = max_iter
        self.minimize = minimize


        # Initialize positions and other variables:
        # - lb and ub: Lower and upper bounds for the search space.
        # - positions: Randomly initialized whale positions within the bounds, each position is a vector of length `dim`.
        # - best_fitness: Tracks the best fitness value found during optimization.
        # - best_solution: Stores the individual corresponding to the best fitness.
        # - convergence_curve: Records the best fitness value at each iteration.
        self.lb = cp.float64(objective_func.lb)
        self.ub = cp.float64(objective_func.ub)
        self.positions = self.__initialize_population()
        self.convergence_curve = cp.zeros(max_iter)
        self.best_fitness = cp.inf if minimize else -cp.inf
        self.best_solution = cp.zeros(self.dim, dtype=cp.float64)

    def __initialize_population(self):
        """Initialize population within bounds."""
        return cp.random.uniform(self.lb, self.ub, (self.pop_size, self.dim), dtype=cp.float64)
   

    def __calculate_fitness(self):
        """Calculate fitness for all whales and update the leader."""

        fitness = cp.zeros(self.pop_size, dtype=cp.float64)

        threads_per_block = 256
        blocks_per_grid = (self.pop_size + threads_per_block - 1) // threads_per_block
        fitness_kernel = self.objective_func.evaluate_on_gpu()

        # Evaluate fitness using the fitness kernel
        fitness_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (
                self.positions,  
                fitness,
                self.pop_size,
                self.dim
            )
        )
        

        # Find the best solution and fitness
        best_idx = cp.argmin(fitness) if self.minimize else cp.argmax(fitness)
        self.best_fitness = fitness[best_idx]
        self.best_solution = self.positions[best_idx].copy()        

    def __update_positions(self, a):
        """Update whale positions based on WOA equations."""
        
        kernel_code = r"""
        #include <curand_kernel.h>
        #define M_PI 3.14159265359 
        extern "C" __global__ void update_positions(
            double* positions, const double* best_solution, const double lb, const double ub,
            double a, int pop_size, int dim, unsigned int seed
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= pop_size) return;
            
            curandState state;
            curand_init(seed, idx, 0, &state);


            double r1 = curand_uniform_double(&state);
            double r2 = curand_uniform_double(&state);
            double A = 2 * a * r1 - a;
            double C = 2 * r2;
            double b = 1.0;
            double l = curand_uniform_double(&state) * 2.0 - 1.0;
            double p = curand_uniform_double(&state);     

            for (int j = 0; j < dim; ++j) {


                double D_Leader, D_X_rand, new_pos;
                if (p < 0.5) {
                    if (fabs(A) < 1) {
                        D_Leader = fabs(C * best_solution[j] - positions[idx * dim + j]);
                        new_pos = best_solution[j] - A * D_Leader;
                    } else {
                        int rand_idx = (int)(curand_uniform_double(&state) * pop_size);
                        D_X_rand = fabs(C * positions[rand_idx * dim + j] - positions[idx * dim + j]);
                        new_pos = positions[rand_idx * dim + j] - A * D_X_rand;
                    }
                } else {
                    double distance2Leader = fabs(best_solution[j] - positions[idx * dim + j]);
                    new_pos = distance2Leader * exp(b * l) * cos(2 * M_PI * l) + best_solution[j];
                }

                positions[idx * dim + j] = fmin(fmax(new_pos, lb), ub);
            }
        }
        """

        update_positions_kernel = cp.RawKernel(kernel_code, 'update_positions')

        threads_per_block = 256
        blocks_per_grid = (self.pop_size + threads_per_block - 1) // threads_per_block

        seed = cp.random.randint(1e9) # Random seed
        update_positions_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (
                self.positions,
                self.best_solution,
                self.lb,
                self.ub,
                cp.float64(a),
                self.pop_size,
                self.dim,
                seed  
            )
        )
        self.positions[0] = self.best_solution  

    def run(self):
        """Run the Whale Optimization Algorithm."""

        # Re-initialize variables for multiple runs
        self.positions = self.__initialize_population()
        self.best_fitness = cp.inf if self.minimize else -cp.inf
        self.best_solution = cp.zeros(self.dim, dtype=cp.float64)
        self.convergence_curve = cp.zeros(self.max_iter)

        for iter in range(self.max_iter):
            self.__calculate_fitness()
            self.convergence_curve[iter] = self.best_fitness


            if iter % 10 == 0:
                print(f"Iteration {iter}: Best Fitness {self.best_fitness}")

            a = 2 - iter * (2 / self.max_iter)  # Linearly decreases from 2 to 0
            self.__update_positions(a)

        self.__calculate_fitness()
        return self.best_solution, self.best_fitness
