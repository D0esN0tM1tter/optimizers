from batch_gradient_descent import GradientDescent
import numpy as np
import matplotlib.pyplot as plt 


def visualize_2D_function(f: callable, history: list):
    # Extract x1 and x2 values from history to determine the plot range
    values = np.array( [entry[0] for entry in history]  )
    x_values = values[: , 0] 
    y_values = values[: , 1] 
    
    # Define the range for the grid based on the history
    x_min , x_max = x_values.min() - 1.5 , x_values.max() + 1.5
    y_min , y_max = y_values.min() - 1.5 , y_values.max() + 1.5 
    
    # Create a grid of points
    x = np.linspace(x_min , x_max , 100) 
    y = np.linspace(y_min , y_max , 100) 
    X , Y = np.meshgrid(x , y)

    # Evaluate the function on the grid
    Z = np.array([f([x , y]) for x , y in zip(X.ravel() , Y.ravel())]).reshape(X.shape)

    # Create a filled contour plot
    plt.contourf(X , Y , Z , levels = 50 , cmap = 'magma')
    
    # Plot the optimization path
    plt.scatter(x_values , y_values , color = 'yellow' , s = 5 , label = 'optimization path')
    plt.scatter(x_values[0] , y_values[0] , color = 'green' , s = 100 , label = 'start')
    plt.scatter(x_values[-1] , y_values[-1] , color = 'red' , s = 100 , label = 'start')

    

    # Add label and legend
    plt.xlabel('x') 
    plt.ylabel('y') 
    plt.title('Optimization path : Gradient descent') 
    plt.legend(loc ='best')
    plt.colorbar()
    plt.savefig('gradient_descent.png' , dpi = 300)
    plt.show()
   


def main():
    def f(x):
        # Quadratic term (global minimum) + Sinusoidal term (local minima)
        return x[0] ** 2 + x[1] ** 2 + x[0] * x[1] + 10 * (np.sin(x[0]) ** 2 + np.sin(x[1]) ** 2)

    def gradf(x):
        # Gradient of the function
        grad_x = 2 * x[0] + x[1] + 10 * 2 * np.sin(x[0]) * np.cos(x[0])  # Chain rule for sin^2(x)
        grad_y = 2 * x[1] + x[0] + 10 * 2 * np.sin(x[1]) * np.cos(x[1])  # Chain rule for sin^2(y)
        return np.array([grad_x, grad_y])

    # Random initial guess in a wider range to increase chances of getting stuck in a local minimum
    initial_guess = np.array([-8 , 8])

    # Instantiate the GradientDescent class
    optimizer = GradientDescent(f, gradf, learning_rate=0.01, tolerance=1e-6, max_iterations=1000)

    # Call the optimize method on the instance
    optimal_value = optimizer.optimize(initial_guess=initial_guess)

    # Visualize the 2D function and optimization path
    visualize_2D_function(f, optimizer.history)


if __name__ == '__main__':
    main()




