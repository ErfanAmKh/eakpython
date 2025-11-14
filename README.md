# eakpython
## Multilayer Perceptron (MLP) from Scratch

A pure Python/NumPy implementation of a Multilayer Perceptron, built from the ground up to demystify the core concepts of Neural Networks.

### Overview

This project involves building a Multi-Layer Perceptron network from scratch without using pre-defined machine learning libraries and packages in Python. I believe that by building the model and training it from scratch, one can understand the processes within a neural network more deeply. Concepts like backpropagation, the effects of hyperparameters, and activation functions can be understood more profoundly and better.

The model is demonstrated and evaluated on the classic **Breast Cancer Wisconsin Dataset** from `scikit-learn`.

**Some results for 50 neurons and 0.1 learning rate**
<img src="images/loss.png" alt="Loss function per epoch" width="60%">
<img src="images/acc.png" alt="Accuracy over train and test sets" width="60%">



### Future Improvements 

The next improvements for the code will be:

*   More result visualizations (confusion matrix, etc.)
*   Implementing more than one hidden layer
*   Visualizing the effects of modifying each of the hyperparameters on the result
*   Support for common activation functions

