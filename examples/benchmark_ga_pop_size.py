from algorithms import GeneticAlgorithmCPU, GeneticAlgorithmGPU
from objective_functions import Ackley
from utilities import benchmark_parameter, benchmark_cpu_gpu

ga_cpu = GeneticAlgorithmCPU(objective_func=Ackley(), pop_size=100, dim=30, max_iter=100)
ga_gpu = GeneticAlgorithmGPU(objective_func=Ackley(), pop_size=100, dim=30, max_iter=100)

pop_sizes = [100, 200, 500, 1000, 2000]
benchmark_parameter(ga_cpu, ga_gpu, param_name='pop_size', param_values=pop_sizes)