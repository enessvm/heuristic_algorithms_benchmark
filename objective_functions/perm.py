from objective_functions import ObjectiveFunction
import numpy as np
import cupy as cp

class Perm(ObjectiveFunction):

    def __init__(self, lb, ub):
        if lb is None or ub is None:
            raise ValueError("You must specify both `lb` (lower bound) and `ub` (upper bound) equal to dim.")
        self.lb = lb
        self.ub = ub
        super().__init__(lb=self.lb, ub=self.ub)


    def evaluate_on_cpu(self, x):
        x = np.asarray_chkfinite(x)  # Ensure the input is finite
        n = len(x)  # Dimension of the input vector
        j = np.arange(1., n + 1)  # Generate the index array from 1 to n
        xbyj = np.fabs(x) / j  # Divide each element of x by its index
        
        # Compute the mean squared term for each k
        result = np.mean([
            np.mean((j**k + 0.5) * (xbyj**k - 1))**2
            for k in j / n
        ])
        
        return result

    def evaluate_on_gpu(self):
        return cp.RawKernel(r'''
        extern "C" __global__
        void fitness_kernel(const double* population, double* fitness, int pop_size, int dim) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= pop_size) return;

            const double b = 0.5;  // Fixed value
            double result = 0.0;

            for (int k = 1; k <= dim; k++) {
                double mean_term = 0.0;
                for (int j = 1; j <= dim; j++) {
                    double x = fabs(population[idx * dim + (j - 1)]) / j;
                    mean_term += (pow(j, (double)k / dim) + b) * (pow(x, (double)k / dim) - 1);
                }
                mean_term /= dim;
                result += mean_term * mean_term;
            }

            fitness[idx] = result / dim;
        }
        ''', 'fitness_kernel')