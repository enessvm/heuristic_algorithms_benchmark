from algorithms import GeneticAlgorithmCPU
from objective_functions import Schwefel

objective_function = Schwefel()

ga = GeneticAlgorithmCPU(
    objective_func=objective_function,
    pop_size=100,         
    dim=10,               
    max_iter=500,         
    crossover_prob=0.9,   
    mutation_prob=0.1,    
    elitism_count=2,      
    minimize=True         
)

best_solution, best_fitness = ga.run()

print("\n=== Results ===")
print(f"Best solution: {best_solution}")
print(f"Best fitness: {best_fitness}")