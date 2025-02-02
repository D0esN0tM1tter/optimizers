from gradient_descent import GDOptimizer
import numpy as np
import matplotlib.pyplot as plt


def visualize_2D_function(f: callable,  history : list , optimizer : str = 'Nesterov Accelerated Gradient' ):
    # Extract x1 and x2 values from history to determine the plot range
    values = np.array([entry[0] for entry in history])
    x_values = values[:, 0]
    y_values = values[:, 1]

    # Define the range for the grid based on the history
    x_min, x_max = x_values.min() - 1.5, x_values.max() + 1.5
    y_min, y_max = y_values.min() - 1.5, y_values.max() + 1.5

    # Create a grid of points
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)

    # Evaluate the function on the grid
    Z = np.array([f([x, y]) for x, y in zip(X.ravel(), Y.ravel())]).reshape(X.shape)

    # Create a filled contour plot
    plt.contourf(X, Y, Z, levels=50, cmap='magma')

    # Plot the optimization path
    plt.scatter(x_values, y_values, color='yellow', s=5, label='Optimization Path')
    plt.scatter(x_values[0], y_values[0], color='green', s=100, label='Start')
    plt.scatter(x_values[-1], y_values[-1], color='red', s=100, label='End')

    # Add label and legend
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Optimization path of : {optimizer}')
    plt.legend(loc='best')
    plt.colorbar()



def f(x):
    return x[0] ** 2 + x[1] ** 2 + x[0] * x[1] + np.sin(x[0] * x[1])

def gradf(x):
    # Gradient of the function
    grad_x = 2 * x[0] + x[1] + x[1] * np.cos(x[0] * x[1])  
    grad_y = 2 * x[1] + x[0] + x[0] * np.cos(x[0] * x[1]) 
    return np.array([grad_x, grad_y])


def test_1() : 
    # initial guess :
    x_0 = np.array([10 , 10])

    # classic gradient descent : 
    optimizer_1 = GDOptimizer(
        f= f , 
        grad_f= gradf,
        learning_rate=0.01 ,
        momentum= 0.0 , 
    )

    # gradient descent with momentum
    optimizer_2 = GDOptimizer(
        f= f , 
        grad_f= gradf , 
        learning_rate=0.01 , 
        momentum=0.9
    )

    # NAG : 
    optimizer_3 = GDOptimizer(
        f= f , 
        grad_f= gradf , 
        learning_rate=0.01 , 
        momentum=0.9,
        nesterov=True
    )

    print(f'optimizer 1 starts ....')
    optimizer_1.optimize(x_0) 

    print(f'\noptimizer 2 starts ....')
    optimizer_2.optimize(x_0)

    print(f'\noptimizer 3 starts ....')
    optimizer_3.optimize(x_0)

    optimizer_1.visualize_convergence(optimizer='GD' , color='blue') 
    optimizer_2.visualize_convergence(optimizer='GD momentum' , color = 'orange') 
    optimizer_3.visualize_convergence(optimizer='NAG' , color='green')
    plt.savefig('plots/convergence_comparison.png' , dpi = 300)
    plt.show()


def test_2() : 

    # initial guess :
    x_0 = np.array([10 , 10])

    # classic gradient descent : 
    optimizer = GDOptimizer(
        f= f , 
        grad_f= gradf,
        learning_rate=0.01 ,
        momentum= 0.9 ,
        nesterov=True 
    )

    optimizer.optimize(x_0) 
    visualize_2D_function(f , optimizer.history , optimizer='Nesterov Accelerated Gradient')
    plt.savefig('plots/nag_path.png' , dpi = 300) 
    plt.show()

  




if __name__ == '__main__':
   test_2()