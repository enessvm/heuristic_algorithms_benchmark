
# Heuristic Algorithms Benchmark: CPU and GPU with CuPy

This project implements and benchmarks three heuristic algorithms: Genetic Algorithm (GA), Whale Optimization Algorithm (WOA), and Ant Lion Optimizer (ALO). It provides both CPU and GPU implementations, utilizing CuPy for GPU acceleration. The benchmark includes 10 popular objective functions, such as Ackley, Schwefel, and Sphere, to evaluate algorithm performance across diverse optimization problems.

## Citations

For more details on the algorithms, please refer to the following papers:

1. Genetic Algorithm (GA): Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems: An Introductory Analysis with Applications to Biology, Control, and Artificial Intelligence*. University of Michigan Press.

2. Whale Optimization Algorithm (WOA): Mirjalili, S., & Lewis, A. (2016). The Whale Optimization Algorithm. *Advances in Engineering Software*, 95, 51-67. [DOI: 10.1016/j.advengsoft.2016.01.008](https://doi.org/10.1016/j.advengsoft.2016.01.008)

3. Ant Lion Optimizer (ALO): Mirjalili, S. (2015). The Ant Lion Optimizer. *Advances in Engineering Software*, 83, 80-98. [DOI: 10.1016/j.advengsoft.2015.01.010](https://doi.org/10.1016/j.advengsoft.2015.01.010)

## Installation
The installation process depends on your system's CUDA version

- If CUDA version 12.x:
   ```bash
   pip install -r requirements.txt
   ```
- If CUDA version is between 11.2 and 11.8   
You will need to manually install CuPy to match your CUDA version.
   ```bash
   pip install cupy-cuda11x
   ```
More info: [CuPy Installation](https://docs.cupy.dev/en/stable/install.html#installing-cupy)

## Usage

### Run Main Benchmark Script
You can run the main benchmark script with different algorithms and objective functions. For example:

   ```
python main.py --algorithm ga_gpu --objective sphere --pop_size 500 --dim 30 --max_iter 1000
   ```
To see all available options and parameters, use:
   ```
python main.py --help
   ```
### Run Examples
The `examples` folder contains predefined scripts for running specific algorithms and benchmarks. For instance:

   ```
python -m examples.benchmark_woa
   ```

## Benchmark Results
![Genetic Algorithm with 30 dimension Ackley objective function](Figure_1.png)
_Execution time comparison for Genetic Algorithm on 30-dimensional Ackley function_
| Population Size | CPU Time (s) | GPU Time (s) |
|  :-----:  |  :-----:  |  :-----:  |
| 100  | 0.24  | 0.10  |
| 200  | 0.51  | 0.06  |
| 500  | 1.55  | 0.06  |
| 1000  | 4.02  | 0.09  |
| 2000  | 11.65  | 0.17  |


![Genetic Algoritm Convergence Curve with 1000 population size, 30 dimension Ackley objective dunction](Figure_2.png)
_Convergence curve for 1000 population size on 30-dimensional Ackley function._
## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
