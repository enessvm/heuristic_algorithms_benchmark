import argparse
from algorithms import (
    GeneticAlgorithmCPU,
    GeneticAlgorithmGPU, 
    WhaleOptimizationCPU, 
    WhaleOptimizationGPU, 
    AntLionOptimizationCPU, 
    AntLionOptimizationGPU
)
from objective_functions import (
    Ackley, 
    DixonPrice, 
    Griewank, 
    Perm, 
    QuadraticMaximization,
    Rastrigin,
    Rosenbrock,
    Schwefel,
    Sphere,
    Zakharov
)

def main():
    # Argument parser for CLI inputs
    parser = argparse.ArgumentParser(description="Run heuristic algorithms on CPU or GPU with customizable parameters.")
    
    # General parameters
    parser.add_argument('--algorithm', type=str, required=True, 
                        choices=['ga_cpu', 'ga_gpu', 'woa_cpu', 'woa_gpu', 'alo_cpu', 'alo_gpu'], 
                        help="Algorithm to run: ga_cpu, ga_gpu, woa_cpu, woa_gpu, alo_cpu, alo_gpu")
    parser.add_argument('--objective', type=str, required=True, 
                        choices=['ackley', 'dixonprice', 'griewank', 'perm', 
                                'quadratic_maximization', 'rastrigin', 'rosenbrock', 
                                'schwefel', 'sphere', 'zakharov'],
                        help="Choose the objective function to use.")
    parser.add_argument('--pop_size', type=int, default=1000, help="Population size (default: 1000)")
    parser.add_argument('--dim', type=int, default=30, help="Dimension of the problem (default: 30)")
    parser.add_argument('--max_iter', type=int, default=1000, help="Maximum iterations (default: 1000)")
    parser.add_argument('--minimize', type=bool, default=True, help="Set to False to maximize (default: True)")
    
    # Optional parameters specific to GA
    parser.add_argument('--crossover_prob', type=float, default=1, help="Crossover probability for GA (default: 1)")
    parser.add_argument('--mutation_prob', type=float, default=0.15, help="Mutation probability for GA (default: 0.15)")
    parser.add_argument('--elitism_count', type=int, default=2, help="Number of elites for GA (default: 2)")
    
    # Parameters for specific objective functions
    # Perm objective function requires lower and upper bounds
    parser.add_argument('--lb', type=float, default=None, help="Lower bound (required for Perm function)")
    parser.add_argument('--ub', type=float, default=None, help="Upper bound (required for Perm function)")

    args = parser.parse_args()

    # Map objective functions
    objective_function_map = {
        'ackley': Ackley,
        'dixonprice': DixonPrice,
        'griewank': Griewank,
        'perm': Perm,
        'quadratic_maximization': QuadraticMaximization,
        'rastrigin': Rastrigin,
        'rosenbrock': Rosenbrock,
        'schwefel': Schwefel,
        'sphere': Sphere,
        'zakharov': Zakharov,
    }


    # Initialize objective function
    if args.objective == 'perm':
        if args.lb is None or args.ub is None:
            raise ValueError("Perm function requires both `lb` and `ub` arguments.")
        objective_func = objective_function_map[args.objective](lb=args.lb, ub=args.ub)
    else:
        objective_func = objective_function_map[args.objective]()

    # Map algorithms
    algorithm_map = {
        'ga_cpu': GeneticAlgorithmCPU,
        'ga_gpu': GeneticAlgorithmGPU,
        'woa_cpu': WhaleOptimizationCPU,
        'woa_gpu': WhaleOptimizationGPU,
        'alo_cpu': AntLionOptimizationCPU,
        'alo_gpu': AntLionOptimizationGPU,
    }

    # Initialize algorithm
    AlgorithmClass = algorithm_map[args.algorithm]
    if 'ga' in args.algorithm:  # If it's a Genetic Algorithm
        algorithm = AlgorithmClass(
            objective_func=objective_func,
            pop_size=args.pop_size,
            dim=args.dim,
            max_iter=args.max_iter,
            crossover_prob=args.crossover_prob,
            mutation_prob=args.mutation_prob,
            elitism_count=args.elitism_count,
            minimize=args.minimize
        )
    else:  # For WOA and ALO
        algorithm = AlgorithmClass(
            objective_func=objective_func,
            pop_size=args.pop_size,
            dim=args.dim,
            max_iter=args.max_iter,
            minimize=args.minimize
        )

    # Run the algorithm
    best_solution, best_fitness = algorithm.run()
    print("\n=== Results ===")
    print(f"Best solution: {best_solution}")
    print(f"\nBest fitness: {best_fitness}")

if __name__ == "__main__":
    main()
