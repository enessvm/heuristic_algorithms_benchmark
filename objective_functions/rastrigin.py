from objective_functions import ObjectiveFunction
import numpy as np
import cupy as cp

class Rastrigin(ObjectiveFunction):
    lb = -5.12
    ub = 5.12

    def __init__(self):
        super().__init__(lb=self.lb, ub=self.ub)

    def evaluate_on_cpu(self, x):
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    

    def evaluate_on_gpu(self):
        return cp.RawKernel(r'''
        #define M_PI 3.141592653589793
        extern "C" __global__
        void fitness_kernel(const double* population, double* fitness, int pop_size, int dim) {                            
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= pop_size) return;
                            
            double fit = 10.0 * dim;
            for (int i = 0; i < dim; i++) {
                double x = population[idx * dim + i];
                fit += x * x - 10.0 * cos(2.0 * M_PI * x);
            }
                
            fitness[idx] = fit;  
        }           
        ''', 'fitness_kernel')
