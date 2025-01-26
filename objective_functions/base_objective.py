class ObjectiveFunction:
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub

    def evaluate_on_cpu(self, x):
        """Evaluate the function on the CPU. Must be overridden."""
        raise NotImplementedError("Subclasses must implement evaluate_cpu!")

    def evaluate_on_gpu(self):
        """Return the CUDA kernel. Must be overridden."""
        raise NotImplementedError("Subclasses must implement cuda_kernel!")

    def name(self):
        """Return the name of the function."""
        return self.__class__.__name__  # Default: Class name as the function name