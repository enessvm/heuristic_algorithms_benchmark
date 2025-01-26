import time
import matplotlib.pyplot as plt

def benchmark_cpu_gpu(cpu_instance, gpu_instance):
    """
    Compares the performance of the same algorithm on CPU and GPU.
    """

    cpu_algorithm_name = type(cpu_instance).__name__
    gpu_algorithm_name = type(gpu_instance).__name__
    algo_name = cpu_algorithm_name.replace('CPU', '')

    print(f"\n{cpu_algorithm_name} running")
    start_time_cpu = time.time()
    _, best_fitness_cpu = cpu_instance.run()
    cpu_time = time.time() - start_time_cpu

    print(f"\n{gpu_algorithm_name} running.")
    start_time_gpu = time.time()
    _, best_fitness_gpu = gpu_instance.run()
    gpu_time = time.time() - start_time_gpu

    print("\n=== Results Comparison ===")
    print(f"CPU Execution Time: {cpu_time:.6f} seconds")
    print(f"GPU Execution Time: {gpu_time:.6f} seconds")
    print(f"CPU Best Fitness: {best_fitness_cpu:.6f}")
    print(f"GPU Best Fitness: {best_fitness_gpu:.6f}")
    print(f"Speedup: {cpu_time / gpu_time:.2f}x")

    # Plot convergence curves
    plt.figure(figsize=(10, 6))
    iterations = range(len(cpu_instance.convergence_curve))
    plt.plot(iterations, cpu_instance.convergence_curve, label=f"(CPU)")
    plt.plot(iterations, gpu_instance.convergence_curve.get(), label=f"(GPU)")
    plt.title(f"Convergence Curve - {algo_name} CPU vs GPU")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return cpu_time, gpu_time


def benchmark_parameter(cpu_instance, gpu_instance, param_name, param_values):
    """
    Benchmarks the effect of a parameter (e.g., population size, dimension) on execution time for CPU and GPU implementations.
    """
    cpu_times = []
    gpu_times = []
    cpu_algorithm_name = type(cpu_instance).__name__
    gpu_algorithm_name = type(gpu_instance).__name__
    algo_name = cpu_algorithm_name.replace('CPU', '')
    
    for value in param_values:
        # Update the parameter value
        setattr(cpu_instance, param_name, value)
        setattr(gpu_instance, param_name, value)

        # Run the benchmarking for the updated parameter
        print(f"\n{cpu_algorithm_name} running")
        start_time_cpu = time.time()
        cpu_instance.run()
        cpu_times.append(time.time() - start_time_cpu)

        print(f"\n{gpu_algorithm_name} running")
        start_time_gpu = time.time()
        gpu_instance.run()
        gpu_times.append(time.time() - start_time_gpu)

    print("\n=== Benchmark Results ===\n")
    header = f"{param_name.replace('_', ' ').title():<10} {'CPU Time (s)':<15} {'GPU Time (s)':<15}"
    print(header)
    print("-" * len(header))
    for param, cpu_time, gpu_time in zip(param_values, cpu_times, gpu_times):
        print(f"{param:<10} {cpu_time:<15.2f} {gpu_time:<15.2f}")

    # Plotting the results 
    x = range(len(param_values))  
    bar_width = 0.35  

    plt.figure(figsize=(10, 6))
    plt.bar([pos - bar_width / 2 for pos in x], cpu_times, bar_width, label='CPU Time', color='blue')
    plt.bar([pos + bar_width / 2 for pos in x], gpu_times, bar_width, label='GPU Time', color='orange')
    plt.xlabel(param_name.replace('_', ' ').title())
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'{algo_name}: CPU and GPU Comparison Across {param_name.replace("_", " ").title()}')
    plt.xticks(x, param_values)  
    plt.legend()
    plt.grid(visible=True, which='both', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return cpu_times, gpu_times
