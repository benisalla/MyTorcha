# Simple Fully connected neural network model.
from MyTorcha.Layer import Layer


class SimpleFC:
    """
    Multi-Layer Perceptron (MLP) class for a feedforward neural network model.
    This class represents a neural network with multiple layers.

    Parameters:
    - in_dim (int): The input dimension, i.e., the number of input features.
    - n_units (list of int): List of integers representing the number of units (neurons) in each hidden layer.
    - n_class (int): The number of output classes.

    Example:
    - To create an MLP with input dimension 64, two hidden layers with 128 and 64 units, and 10 output classes:
      mlp = MLP(in_dim=64, n_units=[128, 64], n_class=10) ==> [64, 128, 64, 10] layers' sizes
    """

    def __init__(self, in_dim, n_units, n_class, act_fun="relu", alpha=0.0):
        """
        Initialize the MLP with specified input dimension, hidden units, and output classes.
        """
        self.in_dim = in_dim
        self.n_class = n_class
        self.n_units = n_units
        self.act_fun = act_fun

        layer_sizes = [in_dim] + n_units + [n_class]
        self.layers = [Layer(layer_sizes[i], layer_sizes[i + 1], act_fun=act_fun, alpha=alpha) for i in range(len(n_units) + 1)]

    def __call__(self, x):
        """
        Forward pass through the MLP, applying each layer to the input data.
        """

        for layer in self.layers:
            x = layer(x)
        return x

    def params(self):
        """
        Get a list of all the parameters (weights and biases) from the layers in the MLP.
        """

        return [p for layer in self.layers for p in layer.params()]
