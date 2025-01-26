from algorithms import WhaleOptimizationCPU, WhaleOptimizationGPU
from objective_functions import Ackley
from utilities import benchmark_cpu_gpu

woa_cpu = WhaleOptimizationCPU(objective_func=Ackley(), pop_size=500, dim=30, max_iter=100)
woa_gpu = WhaleOptimizationGPU(objective_func=Ackley(), pop_size=500, dim=30, max_iter=100)

benchmark_cpu_gpu(woa_cpu, woa_gpu)
