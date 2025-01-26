from objective_functions import ObjectiveFunction
import numpy as np
import cupy as cp

class Rosenbrock(ObjectiveFunction):
    lb = -5.0
    ub = 10.0

    def __init__(self):
        super().__init__(lb=self.lb, ub=self.ub)

    def evaluate_on_cpu(self, x):
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    def evaluate_on_gpu(self):
        return cp.RawKernel(r'''
        extern "C" __global__
        void fitness_kernel(const double* population, double* fitness, int pop_size, int dim) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= pop_size) return;

            double sum = 0.0;

            for (int i = 0; i < dim - 1; i++) {
                double xi = population[idx * dim + i];
                double xnext = population[idx * dim + i + 1];
                sum += 100.0 * (xnext - xi * xi) * (xnext - xi * xi) + (1.0 - xi) * (1.0 - xi);
            }

            fitness[idx] = sum;
        }
        ''', 'fitness_kernel')
