import numpy as np 
from numpy.linalg import norm
import matplotlib.pyplot as plt
import seaborn as sns

class GDOptimizer : 

    def __init__(
            
        self, 
        f              : callable, 
        grad_f         : callable, 
        learning_rate  : float = 0.01, 
        momentum       : float = 0.9 , 
        decay          : float = 0.0 , 
        dampening      : float = 0.0 , 
        max_iterations : int   = 1000, 
        tolerance      : float = 1e-6  ,
        nesterov       : bool  = False,
        maximize       : bool  = False

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
        self.momentum = momentum
        self.decay = decay
        self.dampnening = dampening
        self.max_iterations = max_iterations 
        self.tolerance = tolerance
        self.nesterov = nesterov
        self.maximize = maximize
        self.history = list()


    def optimize(self , x_0 : np.ndarray) -> np.ndarray :

        # starting point
        x = np.array(x_0)

        # initial velocity : 
        velocity = np.zeros_like(x) 

        for i in range(self.max_iterations) : 

            # calculate the current gradient : 
            grad = self.grad_f(x)

            # calculate the function at the current point and keeping track of the values  
            f_value = self.f(x) 
            self.history.append((x.copy() , f_value.copy()))

            if i % 100 == 0 :
                print(f'Iteration {i + 1} : x = {np.round(x , 3)} , f = {np.round(f_value , 3)} ')

            # regularization : controlling the magnitude of the parameters 
            if self.decay > 0 : 
                grad = grad + self.decay * x 
            
            # momentum : smoothening the updates 
            if self.momentum > 0 :
                velocity = (self.momentum * velocity) + (1 - self.dampnening) * grad
            
                # Nestorov accelerated gradient
                if self.nesterov : 
                    grad = grad + (self.momentum * velocity)
                
                else : 
                    grad = velocity
            
            if self.maximize :
                x_new = x + (self.learning_rate) * grad
            
            else : 
                x_new = x - (self.learning_rate) * grad
            
            

            if norm(x_new - x) <= self.tolerance : 
                print(f'Converged after {i + 1} iterations')
                break




            x = x_new

        return x 

        


    def visualize_convergence(self , optimizer : str = 'NAG' , color : str = 'blue') -> None : 

        values = np.array([entry[1] for entry in self.history])
        sns.set_style('darkgrid')
        plt.plot(values  ,  label = optimizer , color = color ) 
        plt.xlabel('iterations') 
        plt.ylabel('function value') 
        plt.title(f'Convergence Plot') 
        plt.legend(loc = 'best')




    