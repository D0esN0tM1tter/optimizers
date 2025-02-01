### Introduction  
This repository contains implementations of some of the most widely used optimization algorithms in AI and Deep Learning. These algorithms are developed from scratch and categorized into three main sections: Gradient Descent Variants, Adaptive Learning Algorithms, and Second-Order Methods.  

### Gradient Descent Variants  
Gradient descent is a fundamental optimization technique used to minimize loss functions. These variants improve convergence speed and stability.  

- **Ordinary Gradient Descent** – The basic form of gradient descent that updates parameters in the direction of the steepest descent.  
- **Gradient Descent with Momentum** – Introduces momentum to accelerate updates and reduce oscillations.  
- **Nesterov Accelerated Gradient** – An improvement over momentum-based descent, predicting future updates for better convergence.  

### Adaptive Learning Algorithms  
These methods dynamically adjust learning rates based on past gradients to improve training efficiency.  

- **AdaGrad** – Adapts the learning rate individually for each parameter based on historical gradients.  
- **RMSProp** – Uses an exponentially decaying average of past squared gradients to maintain a stable learning rate.  
- **AdaDelta** – An extension of RMSProp that removes the need for manually selecting an initial learning rate.  
- **Adam** – Combines momentum and adaptive learning rates to achieve fast and stable convergence.  

### Second-Order Methods  
Unlike first-order methods, these algorithms use second-order derivatives (Hessian or approximations) for more accurate optimization steps.  

- **Newton's Method** – Uses second-order derivative information to find optimal points faster, but requires computing the Hessian matrix.  
- **L-BFGS Method** – A memory-efficient approximation of second-order optimization, widely used for large-scale problems.  
- **Natural Gradient Descent** – Adjusts the update direction based on the geometry of the parameter space for more efficient learning.  
