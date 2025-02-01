import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import seaborn as sns

class RMSProp:
    def __init__(
            self, 
            f: callable, 
            grad_f: callable, 
            learning_rate: float = 0.01, 
            momentum: float = 0.9, 
            max_iters: int = 1000,
            tolerance: float = 1e-6
    ) -> None:
        '''
        f             : objective function 
        grad_f        : gradient of the objective function 
        learning_rate : step size for parameter update 
        momentum      : weight for the past gradients 
        max_iters     : maximum iterations 
        tolerance     : smallest difference allowed between two consecutive updates
        '''
        self.f = f
        self.grad_f = grad_f
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.history = []

    def optimize(self, initial_guess: np.ndarray) -> np.ndarray:
        # Initialize parameters
        x = np.array(initial_guess, dtype=np.float64)
        velocity = np.zeros_like(x)
        epsilon = 1e-8  # Small constant to prevent division by zero

        for i in range(self.max_iters):
            # Evaluate the function and its gradient
            value = self.f(x)
            grad = self.grad_f(x)

            self.history.append((x.copy(), value))  # Fix copy() issue

            if i % 100 == 0:
                print(f'Iteration {i}/{self.max_iters}: x = {x}, f = {value}')

            # Update velocity (exponential moving average of squared gradients)
            velocity = self.momentum * velocity + (1 - self.momentum) * grad**2

            # Update parameter
            grad_adjusted = grad / (np.sqrt(velocity + epsilon))  # Adjust gradient by RMSProp scaling
            x_new = x - self.learning_rate * grad_adjusted

            # Check for convergence (both parameter and function value)
            if norm(x_new - x) <= self.tolerance and abs(value - self.history[-2][1]) <= self.tolerance:
                print(f'Converged after {i} iterations.')
                return x_new  # Return early if converged

            x = x_new

        return x  # Return the final value after max_iters

    def visualize_convergence(self) -> None:
        values = np.array([entry[1] for entry in self.history])
        sns.set_style('darkgrid')
        plt.plot(values, color='navy', label='Convergence')
        plt.xlabel('Iterations') 
        plt.ylabel('Function value') 
        plt.title('Convergence plot: RMSProp') 
        plt.show()
