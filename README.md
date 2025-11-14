# eakpython
## Multilayer Perceptron (MLP) from Scratch

A pure Python/NumPy implementation of a Multilayer Perceptron, built from the ground up to demystify the core concepts of deep learning.

### Overview

This project implements a Multilayer Perceptron (MLP) entirely from scratch, using only fundamental Python libraries. The deliberate avoidance of high-level machine learning frameworks (such as TensorFlow or PyTorch) provides a deep, hands-on understanding of the core mechanics behind neural networks.

By building the model, training loop, and backpropagation algorithm manually, this project offers invaluable insight into fundamental concepts like:
*   **Backpropagation** and gradient computation.
*   The role and impact of **activation functions**.
*   How **hyperparameters** (learning rate, epochs, etc.) influence training and performance.

The model is demonstrated and evaluated on the classic **Breast Cancer Wisconsin Dataset** from `scikit-learn`.

### Features

*   Custom implementation of forward and backward propagation.
*   Configurable network architecture (number of nodes per layer).
*   Support for common activation functions (Sigmoid, ReLU, Tanh).
*   Standard optimization using Gradient Descent.

### Future Improvements & Roadmap

This is an ongoing project for learning and experimentation. Planned enhancements include:

*   **Enhanced Visualization:** Implementing plots for training loss/accuracy and a confusion matrix to better evaluate model performance.
*   **Architectural Depth:** Extending the network to support more than one hidden layer.
*   **Hyperparameter Analysis:** Creating visualizations to demonstrate the effects of modifying hyperparameters (e.g., learning rate, network architecture) on the final results.

