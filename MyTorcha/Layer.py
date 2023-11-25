from MyTorcha.Neuron import Neuron


class Layer:

    def __init__(self, in_size, units, act_fun="relu", alpha=0.0):
        """
        Initialize a Layer with specified parameters.
        """
        self.in_size = in_size
        self.units = units
        self.act_fun = act_fun

        self.neurons = [Neuron(in_size, act_fun=act_fun, alpha=alpha) for _ in range(units)]

    def __call__(self, x):
        """
        Compute the output of the entire layer for a given input.
        """
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def params(self):
        """
        Get a list of layer's neurons parameters: weights and bias.
        """
        return [p for neuron in self.neurons for p in neuron.params()]
