from objective_functions import ObjectiveFunction
import numpy as np
import cupy as cp

class Ackley(ObjectiveFunction):
    lb = -32.768
    ub = 32.768

    def __init__(self):
        super().__init__(lb=self.lb, ub=self.ub)

    def evaluate_on_cpu(self, x):
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
        term2 = -np.exp(sum2 / d)
        return term1 + term2 + a + np.exp(1)

    def evaluate_on_gpu(self):
        return cp.RawKernel(r'''
        #define M_PI 3.141592653589793
        extern "C" __global__
        void fitness_kernel(const double* population, double* fitness, int pop_size, int dim) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= pop_size) return;
                            
            const double a = 20.0;
            const double b = 0.2;
            const double c = 2.0 * M_PI;
            
            double sum1 = 0.0;
            double sum2 = 0.0;
            
            for (int i = 0; i < dim; i++) {
                double x = population[idx * dim + i];
                sum1 += x * x;
                sum2 += cos(c * x);
            }
            
            double term1 = -a * exp(-b * sqrt(sum1 / dim));
            double term2 = -exp(sum2 / dim);
            
            fitness[idx] = term1 + term2 + a + exp(1.0);
 
        }
        ''', 'fitness_kernel')
