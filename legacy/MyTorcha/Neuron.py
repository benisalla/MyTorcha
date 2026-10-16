from Helpers.helpers import sum_nodes
from MyTorcha.Node import Node
import random


class Neuron:
    """
    A class representing a single neuron in a neural network.

    Args:
        in_size (int): The number of input features.
        act_fun (str, optional): The activation function to use ('tanh', 'sigmoid', or 'relu'). Default is 'relu'.

    Attributes:
        weights (list): List of weights associated with input features.
        bias (Node): The bias term.
        act_fun (str): The activation function used by the neuron.

    Methods:
        parameters(): Get a list of neuron parameters (weights and bias).
    """

    def __init__(self, in_size, act_fun='relu', alpha=0.0):
        """
        Initialize a neuron with random weights and a bias term.
        """
        self.in_size = in_size
        self.act_fun = act_fun
        self.w = [Node(random.uniform(-1, 1), _alpha=alpha, _nm="w") for _ in range(in_size)]
        self.b = Node(random.uniform(-1, 1), _alpha=alpha, _nm="b")

        self.act_fun = lambda: None

        # build activation function
        self.__init_act_fun__(act_fun)

    def __init_act_fun__(self, act_fun):

        assert act_fun in ("elu",
                           "gelu",
                           "linear",
                           "leaky_relu",
                           "softplus",
                           "sigmoid",
                           "relu",
                           "tanh",
                           "arctan"), f"({act_fun})! there is no such a function !"

        if act_fun == 'tanh':
            def activation(x):
                return x.tanh()

            self.act_fun = activation

        elif act_fun == 'elu':
            def activation(x):
                return x.elu()

            self.act_fun = activation

        elif act_fun == 'sigmoid':
            def activation(x):
                return x.sigmoid()

            self.act_fun = activation

        elif act_fun == 'leaky_relu':
            def activation(x):
                return x.leaky_relu()

            self.act_fun = activation

        elif act_fun == 'softplus':
            def activation(x):
                return x.softplus()

            self.act_fun = activation

        elif act_fun == 'arctan':
            def activation(x):
                return x.arctan()

            self.act_fun = activation

        elif act_fun == 'linear':
            def activation(x):
                return x.linear()

            self.act_fun = activation

        elif act_fun == 'gelu':
            def activation(x):
                return x.gelu()

            self.act_fun = activation

        else:  # relu by default
            def activation(x):
                return x.relu()

            self.act_fun = activation

    def __call__(self, inputs):
        """
        Compute the output of the neuron for a given input.
        """
        act = sum_nodes([wi * xi for wi, xi in zip(self.w, inputs)]) + self.b
        return self.act_fun(act)

    def params(self):
        """
        Get a list of neuron parameters: weights and bias.
        """
        return self.w + [self.b]
