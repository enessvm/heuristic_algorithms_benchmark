from objective_functions import ObjectiveFunction
import numpy as np
import cupy as cp

class DixonPrice(ObjectiveFunction):
    lb = -10.0
    ub = 10.0

    def __init__(self):
        super().__init__(lb=self.lb, ub=self.ub)

    def evaluate_on_cpu(self, x):
        term1 = (x[0] - 1)**2
        term2 = np.sum([(i + 1) * (2 * x[i]**2 - x[i - 1])**2 for i in range(1, len(x))])
        return term1 + term2

    def evaluate_on_gpu(self):
        return cp.RawKernel(r'''
        extern "C" __global__
        void fitness_kernel(const double* population, double* fitness, int pop_size, int dim) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= pop_size) return;

            double term1 = (population[idx * dim] - 1.0) * (population[idx * dim] - 1.0);
            double term2 = 0.0;

            for (int i = 1; i < dim; i++) {
                double xi = population[idx * dim + i];
                double xprev = population[idx * dim + i - 1];
                term2 += (i + 1) * (2.0 * xi * xi - xprev) * (2.0 * xi * xi - xprev);
            }

            fitness[idx] = term1 + term2;
        }
        ''', 'fitness_kernel')
