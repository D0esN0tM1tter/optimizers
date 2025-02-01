import numpy as np 
from numpy.linalg import norm
import matplotlib.pyplot as plt

# Batch gradient descent class 
class GradientDescent : 

    def __init__(
            
        self, 
        f  : callable, 
        grad_f : callable, 
        learning_rate : float = 0.01, 
        max_iterations : int = 1000, 
        tolerance : float = 1e-6

    ) -> None :
        
        '''
        Batch gradient descent optimizer

        Parameters : 
        - f              : objective function 
        - grad_f         : gradient of the objective function 
        - learning_rate  : step size for each update 
        - max_iterations : maximum iterations before stopping 
        - tolerance      : convergence criterion
        '''

        self.f = f
        self.grad_f = grad_f 
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations 
        self.tolerance = tolerance
        self.history = list()


    def optimize(self , initial_guess : np.ndarray) -> np.ndarray :

        '''
        Runs the gradient descent optimization algorithm : 

        parameters : 
        - initial_guess : starting point for the algorithm

        returns : 
        - optimal value

        '''

        x = initial_guess 

        for i in range(self.max_iterations) : 

            current_value = self.f(x) 
            current_grad = self.grad_f(x) 

            self.history.append((x.copy() , current_value.copy()))

            # Logging every 100 iterations : 
            if i % 100 == 0 : 
                print(f'Iterations {i + 1} : x = {x} , f = {np.round(current_value , 3)}')

            # Gradient step : 
            x_new = x - self.learning_rate * current_grad

            # Relative convergence check : 
            if norm(x_new - x) / ( norm(x) + 1e-10) <= self.tolerance : 
                print(f'converged after {i + 1} iterations')
                break
            
            x = x_new
        
        return x


    def visualize_convergence(self) -> None : 

        values = np.array([entry[1] for entry in self.history])

        plt.plot(values ,  color = 'orange' , label = 'convergence') 
        plt.xlabel('Iterations') 
        plt.ylabel('Function value') 
        plt.title('Convergence plot') 
        plt.show()




    