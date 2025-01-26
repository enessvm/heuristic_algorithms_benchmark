from algorithms import AntLionOptimizationGPU
from objective_functions import Ackley

objective_function = Ackley()

alo = AntLionOptimizationGPU(
    objective_func=objective_function,
    pop_size=1000,         
    dim=30,               
    max_iter=500,               
)

best_solution, best_fitness = alo.run()

print("\n=== Results ===")
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")