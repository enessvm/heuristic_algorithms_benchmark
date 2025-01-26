from objective_functions import ObjectiveFunction
import numpy as np
import cupy as cp

class Schwefel(ObjectiveFunction):
    lb = -500
    ub = 500

    def __init__(self):
        super().__init__(lb=self.lb, ub=self.ub)

    def evaluate_on_cpu(self, x):
        d = len(x)
        sum_term = np.sum(x * np.sin(np.sqrt(np.abs(x))))
        return 418.9829 * d - sum_term

    def evaluate_on_gpu(self):
        return cp.RawKernel(r'''
        extern "C" __global__
        void fitness_kernel(const double* population, double* fitness, int pop_size, int dim) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx > pop_size) return;
                            
            double sum_term = 0.0;
            for (int i = 0; i < dim; i++) {
                double x = population[idx * dim + i];
                sum_term += x * sin(sqrt(fabs(x)));
            }
            fitness[idx] = 418.9829 * dim - sum_term;

        }
        ''', 'fitness_kernel')
