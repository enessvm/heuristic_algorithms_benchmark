from objective_functions import ObjectiveFunction
import numpy as np
import cupy as cp

class Zakharov(ObjectiveFunction):
    lb = -5.0
    ub = 10.0

    def __init__(self):
        super().__init__(lb=self.lb, ub=self.ub)

    def evaluate_on_cpu(self, x):
        d = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(0.5 * (np.arange(1, d + 1)) * x)
        return sum1 + sum2**2 + sum2**4

    def evaluate_on_gpu(self):
        return cp.RawKernel(r'''
        extern "C" __global__
        void fitness_kernel(const double* population, double* fitness, int pop_size, int dim) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= pop_size) return;

            double sum1 = 0.0;
            double sum2 = 0.0;

            for (int i = 0; i < dim; i++) {
                double x = population[idx * dim + i];
                sum1 += x * x;
                sum2 += 0.5 * (i + 1) * x;
            }

            fitness[idx] = sum1 + (sum2 * sum2) + (sum2 * sum2 * sum2 * sum2);
        }
        ''', 'fitness_kernel')