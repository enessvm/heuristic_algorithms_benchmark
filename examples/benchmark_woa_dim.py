from algorithms import WhaleOptimizationCPU, WhaleOptimizationGPU
from objective_functions import Ackley
from utilities import benchmark_parameter

woa_cpu = WhaleOptimizationCPU(objective_func=Ackley(), pop_size=1000, dim=30, max_iter=100)
woa_gpu = WhaleOptimizationGPU(objective_func=Ackley(), pop_size=1000, dim=30, max_iter=100)

dims = [5, 20, 50, 100]

benchmark_parameter(woa_cpu, woa_gpu, param_name='dim', param_values=dims)
