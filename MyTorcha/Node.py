import math
from graphviz import Digraph


# The building block of the entire Neural Nets
class Node:
    """
    Node class represents a node in a computational graph for neural networks.
    It encapsulates a numerical value and provides methods for building and
    manipulating the graph, as well as computing gradients during backpropagation.
    """

    def __init__(self, _vl, _alpha=0.0, _ins=(), _op='', _nm=''):
        """
        Args:
        _vl (float): node's value.
        _ins (tuple): Inputs nodes connected to this node.
        _op (str): The operation that created this node.
        _nm (str): The name for the node.
        """
        self._vl = _vl
        self._gd = 0.0
        self._op = _op
        self._nm = _nm
        self._ins = set(_ins)
        self._alpha = _alpha
        self._bck = lambda: None

    def __repr__(self):
        return f"{self._vl}"

    # Overloaded operators for arithmetic operations

    # self + right_node | number
    def __add__(self, r_node):
        """
        Args:
        r_node (Node or float): The right node or a constant to add.
        """
        r_node = r_node if isinstance(r_node, Node) else Node(r_node)
        out = Node(self._vl + r_node._vl, _alpha=self._alpha, _ins=(self, r_node), _op='+')

        def _bck():
            self._gd += 1.0 * out._gd
            r_node._gd += 1.0 * out._gd

        out._bck = _bck

        return out

    # self * right_node | number
    def __mul__(self, r_node):
        """
        Args:
        r_node (Node or float): The right node or a constant to multiply.
        """
        r_node = r_node if isinstance(r_node, Node) else Node(r_node)
        out = Node(self._vl * r_node._vl, _alpha=self._alpha, _ins=(self, r_node), _op='*')

        def _bck():
            self._gd += r_node._vl * out._gd
            r_node._gd += self._vl * out._gd

        out._bck = _bck

        return out

    # self ** power(Node | number)
    def __pow__(self, power):
        """
        Args:
        power (Node or float): The power to raise the node to.
        """
        out = None
        if isinstance(power, Node):
            out = Node(self._vl ** power._vl, _alpha=self._alpha, _ins=(self,), _op=f'**{power._vl}')

            def _bck():
                self._gd += power._vl * (self._vl ** (power._vl - 1)) * out._gd

        elif isinstance(power, (float, int)):
            out = Node(self._vl ** power, _alpha=self._alpha, _ins=(self,), _op=f'**{power}')

            def _bck():
                self._gd += power * (self._vl ** (power - 1)) * out._gd

        else:
            print("We do not support this opperation for the moment !")

        out._bck = _bck

        return out

    # right_node * self
    def __rmul__(self, l_node):
        return self * l_node

    # self / right_node
    def __truediv__(self, r_node):
        return self * (r_node ** -1)

    # -self
    def __neg__(self):
        return self * Node(-1)

    # self - right_node
    def __sub__(self, r_node):
        return self + (-r_node)

    # left_node + self
    def __radd__(self, l_node):
        return self + l_node

    # left_node - self
    def __rsub__(self, l_node):
        return l_node + (-self)

    # abs(node)
    def __abs__(self):
        return self if self._vl > 0 else -self

    # copy
    def copy(self):
      new_node = Node( _vl=self._vl,
                      _alpha=self._alpha,
                      _op=self._op,
                      _nm=self._nm)
      new_node._gd = self._gd
      new_node._bck = self._bck
      new_node._ins = set(self._ins)

      return new_node

    # hash ( very useful :) )
    def __hash__(self):
        if self._ins is None:
            return hash((self._vl, self._alpha, self._op, self._nm))

        # Hash each element in _ins using their unique IDs
        ins_hashes = tuple(hash((nd._vl, nd._gd, nd._op, nd._nm)) for nd in self._ins)
        return hash((self._vl, self._alpha, ins_hashes, self._op, self._nm))

    # ==
    def __eq__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return self._vl == other._vl

    # <
    def __lt__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return self._vl < other._vl

    # <=
    def __le__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return self._vl <= other._vl

    # >
    def __gt__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return self._vl > other._vl

    # >=
    def __ge__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        return self._vl >= other._vl

    # build activation functions

    # tanh
    def tanh(self):
        """
        Compute the activation function tanh.
        """
        out = Node((math.exp(2 * self._vl) - 1) / (math.exp(2 * self._vl) + 1), _alpha=self._alpha, _ins=(self,),
                   _op='tanh')

        def _bck():
            self._gd += (1 - out._vl ** 2) * out._gd

        out._bck = _bck

        return out

    # exp
    def exp(self):
        out = Node(math.exp(self._vl), _alpha=self._alpha, _ins=(self,), _op='exp')

        def _bck():
            self._gd += out._vl * out._gd

        out._bck = _bck

        return out

    # log
    def log(self):
        out = Node(math.log(self._vl), _alpha=self._alpha, _ins=(self,), _op='log')

        def _bck():
            self._gd += out._gd / (out._vl + 1e-7)  # 1e-7 for numerical safety !

        out._bck = _bck

        return out

    # sigmoid
    def sigmoid(self):
        """
        Compute the activation function: sigmoid.
        """
        out = Node((1 / (1 + math.exp(-self._vl))), _alpha=self._alpha, _ins=(self,), _op='sigmoid')

        def _bck():
            self._gd += out._vl * (1 - out._vl) * out._gd

        out._bck = _bck

        return out

    # relu
    def relu(self):
        """
        Compute the activation function: relu.
        """
        x = self._vl if self._vl > 0 else 0
        out = Node(x, _alpha=self._alpha, _ins=(self,), _op='relu')

        def _bck():
            self._gd += out._gd if out._vl > 0.0 else 0.0

        out._bck = _bck

        return out

    # leaky_relu
    def leaky_relu(self):
        """
        Compute the activation function: learky relu.
        """
        x = self._vl if self._vl > 0 else self._alpha * self._vl
        out = Node(x, _alpha=self._alpha, _ins=(self,), _op='leaky_relu')

        def _bck():
            self._gd += out._gd if out._vl > 0 else self._alpha * out._gd

        out._bck = _bck

        return out

    # linear
    def linear(self):
        """
        Compute the activation function: linear.
        """
        out = Node(self._vl, _alpha=self._alpha, _ins=(self,), _op='linear')

        def _bck():
            self._gd += out._gd

        out._bck = _bck

        return out

    # arctan
    def arctan(self):
        """
        Compute the activation function: arctan.
        """
        x = math.atan(self._vl)
        out = Node(x, _alpha=self._alpha, _ins=(self,), _op='arctan')

        def _bck():
            self._gd += out._gd * (1.0 / (self._vl ** 2 + 1.0))

        out._bck = _bck

        return out

    # elu
    def elu(self):
        """
        Compute the activation function: elu.
        """
        x = self._alpha * (math.exp(self._vl) - 1) if self._vl < 0 else self._vl
        out = Node(x, _alpha=self._alpha, _ins=(self,), _op='elu')

        def _bck():
            self._gd += out._gd if self._vl > 0 else out._gd * self._vl * math.exp(self._vl)

        out._bck = _bck

        return out

    # softplus
    def softplus(self):
        """
        Compute the activation function: softplus.
        """
        x = math.log(math.exp(self._vl) + 1)
        out = Node(x, _alpha=self._alpha, _ins=(self,), _op='softplus')

        def _bck():
            self._gd += out._gd * (1 / (1 + math.exp(-self._vl)))

        out._bck = _bck

        return out

    # gelu
    def gelu(self):
        """
        Compute the activation function: gelu.
        """
        x = 0.5 * self._vl * (1 + math.tanh(math.sqrt(2 / math.pi) * (self._vl + 0.044715 * self._vl ** 3)))
        out = Node(x, _alpha=self._alpha, _ins=(self,), _op='gelu')

        def _bck():
            x = self._vl
            com = math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)
            sech = (2 / (math.exp(com) + math.exp(-com))) ** 2
            b = math.sqrt(2 / math.pi) * (1 + 3 * 0.044715 * x ** 2)
            self._gd += out._gd * (0.5 * (1 + math.tanh(com)) + 0.5 * x * sech * b)

        out._bck = _bck

        return out

    # Now let's handle the graph (backprop, display, ...)

    # backpropagation
    def backward(self):
        """
        Perform backpropagation to compute gradients for this node and its ancestors.
        """
        visited = set()
        topology = []

        def make_topology(node):
            if node not in visited:
                visited.add(node)
                for prev in node._ins:
                    make_topology(prev)
                topology.append(node)

        make_topology(self)

        # self._gd = 1.0
        for node in reversed(topology):
            node._bck()

    # show history of current Node
    def draw(head):
        """
        Visualize the computational graph starting from the given node.

        Args:
        head (Node): The head node from which to visualize the graph.

        Returns:
        Digraph: A visualization of the computational graph.
        """
        dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

        def scan_graph(head):
            nodes, edges = set(), set()

            def make(n):
                if n not in nodes:
                    nodes.add(n)
                    for prev in n._ins:
                        edges.add((prev, n))
                        make(prev)

            make(head)
            return nodes, edges

        nodes, edges = scan_graph(head)
        for n in nodes:
            uid = str(id(n))
            dot.node(name=uid, label="{ %s | val(%.4f) | grad(%.4f) }" % (n._nm, n._vl, n._gd), shape='record')
            if n._op:
                dot.node(name=uid + n._op, label=n._op, shape='circle')
                dot.edge(uid + n._op, uid)

        for n1, n2 in edges:
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)

        return dot
