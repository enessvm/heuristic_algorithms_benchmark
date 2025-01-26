from objective_functions import ObjectiveFunction
import numpy as np
import cupy as cp

class Griewank(ObjectiveFunction):
    lb = -600
    ub = 600

    def __init__(self):
        super().__init__(lb=self.lb, ub=self.ub)

    def evaluate_on_cpu(self, x):
        sum_part = np.sum(x**2) / 4000
        prod_part = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return sum_part - prod_part + 1

    def evaluate_on_gpu(self):
        return cp.RawKernel(r'''
        extern "C" __global__
        void fitness_kernel(const double* population, double* fitness, int pop_size, int dim) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= pop_size) return;

            double sum_part = 0.0;
            double prod_part = 1.0;
            for (int i = 0; i < dim; i++) {
                double x = population[idx * dim + i];
                sum_part += x * x / 4000.0;
                prod_part *= cos(x / sqrt((double)(i + 1)));
            }

            fitness[idx] = sum_part - prod_part + 1.0;
        }
        ''', 'fitness_kernel')
