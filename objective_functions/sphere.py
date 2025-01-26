from objective_functions import ObjectiveFunction
import numpy as np
import cupy as cp

class Sphere(ObjectiveFunction):
    lb = -5.12
    ub = 5.12

    def __init__(self):
        super().__init__(lb=self.lb, ub=self.ub)

    def evaluate_on_cpu(self, x):
        return np.sum(x**2)

    def evaluate_on_gpu(self):
        return cp.RawKernel(r'''
        extern "C" __global__
        void fitness_kernel(const double* population, double* fitness, int pop_size, int dim) {                            
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= pop_size) return;
                            
            double fit = 0.0;
            for (int i = 0; i < dim; i++) {
                double x = population[idx * dim + i];
                fit += x * x;
            }
                
            fitness[idx] = fit;  
        }           
        ''', 'fitness_kernel')
