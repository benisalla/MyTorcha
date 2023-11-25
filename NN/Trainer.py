import math
import numpy as np
from Helpers.helpers import sum_nodes


# Note: This code is not specific to any neural network architecture;
# it can be used to train various models.
class Trainer:

    def __init__(self,
                 model,
                 lr=0.01,
                 batch_size=8,
                 loss="mae",
                 optimizer="gradient_descent",
                 beta1=0.99,
                 beta2=0.994,
                 mu=0.9,
                 decay_rate=0.9):
        self.model = model
        self.lr = lr
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.mu = mu
        self.decay_rate = decay_rate
        self.loss = lambda: None
        self.optimizer = lambda: None

        self.__init_optimizer__(optimizer)
        self.__init_loss_func__(loss)

    def __init_loss_func__(self, loss):
        if loss.lower() == "mse":
            def loss_func(logits, true_y):
                loss_value = sum_nodes([(pred_y - y) ** 2 for pred_y, y in zip(logits, true_y)])
                loss_value._gd = 1.0
                return loss_value, None

            self.loss = loss_func

        elif loss.lower() == "mae":
            def loss_func(logits, true_y):
                loss_value = sum_nodes([abs(pred_y - y) for pred_y, y in zip(logits, true_y)])
                loss_value._gd = 1.0
                return loss_value, None

            self.loss = loss_func

        elif loss.lower() == "cross_entropy":
            def loss_func(logits, true_y):
                # softmax (Aaaa..aa it is a monster)
                sum = sum_nodes([x.exp() for x in logits])
                probs = [x.exp() / sum for x in logits]

                oh_y = [0] * self.model.n_class
                oh_y[true_y] = 1

                loss_value = -probs[true_y].log()

                # Little hack to softmax
                for i, logit in enumerate(logits):
                    logit._gd = (probs[i] - oh_y[i])._vl

                # nice hack Mr. backward :)
                return loss_value, logits

            self.loss = loss_func

        else:
            print("There is no such loss function !")

    def __init_optimizer__(self, opt):
        assert opt in ("adam",
                       "momentum_sgd",
                       "nesterov_momentum",
                       "adagrad",
                       "rms_prop",
                       "gradient_descent",), f"({opt})! there is no such optimizer !"

        # Adam optimizer
        if opt == "adam":
            self.momentum = 0
            self.decay = 0

            def opt_function():
                for p in self.model.params():
                    self.momentum = self.beta1 * self.momentum + (1 - self.beta1) * p._gd
                    self.decay = self.beta2 * self.decay + (1 - self.beta2) * (p._gd ** 2)
                    p._vl += -self.lr * self.momentum / (math.sqrt(self.decay) + 1e-7)

            self.optimizer = opt_function

        # gradient descent with momentum optimizer
        elif opt == "momentum_sgd":
            self.momentum = 0

            def opt_function():
                for p in self.model.params():
                    self.momentum = self.mu * self.momentum - self.lr * p._gd
                    p._vl += self.momentum

            self.optimizer = opt_function

        # Nesterv Momentum optimizer
        elif opt == "nesterov_momentum":
            self.momentum = 0

            def opt_function():
                for p in self.model.params():
                    prev = self.momentum
                    self.momentum = self.mu * self.momentum - self.lr * p._gd
                    p._vl += -self.mu * prev + (1 + self.mu) * self.momentum

            self.optimizer = opt_function

        # AdaGrad optimizer
        elif opt == "adagrad":
            self.cache = 0

            def opt_function():
                for p in self.model.params():
                    self.cache += p._gd ** 2
                    p._vl += -self.lr * p._gd / (math.sqrt(self.cache) + 1e-7)

            self.optimizer = opt_function

        # RMSprop optimizer
        elif opt == "rms_prop":
            self.cache = 0

            def opt_function():
                for p in self.model.params():
                    self.cache = self.decay_rate * self.cache + (1 - self.decay_rate) * p._gd ** 2
                    p._vl += -self.lr * p._gd / (math.sqrt(self.cache) + 1e-7)

            self.optimizer = opt_function

        # gradient descent optimizer
        elif opt == "gradient_descent":
            def opt_function(model):
                print("we are in sgd")
                for p in model.params():
                    p._vl -= self.lr * p._gd
                return model

            self.optimizer = opt_function

        else:
            print("There is no such optimizer !")

    def zero_grad(self):
        for p in self.model.params():
            p._gd = 0.0

    def backprop(self, loss):
        if isinstance(loss, tuple):
            _, logits, true_y = loss
            for i, logit in enumerate(logits):
                logit.backward()
        else:
            loss.backward()

    def compute(self, xs, ys):
        losses = []

        # clean the graph
        self.zero_grad()

        for x, true_y in zip(xs, ys):
            logits = self.model(x)

            # compute loss fun (I mean function hhh)
            loss, _ = self.loss(logits=logits, true_y=true_y)
            losses.append(loss)

            # backprop
            self.backprop(loss)
        return losses

    def fit(self, inputs, true_ys, epochs=10):
        tsize = len(true_ys)
        for k in range(epochs):
            # pick random batch each time
            ix = np.random.randint(0, tsize - self.batch_size)
            xb = inputs[ix:ix + self.batch_size, :]
            yb = true_ys[ix:ix + self.batch_size]

            # compute batch
            losses = self.compute(xb, yb)

            # now update
            self.optimizer()

            # display loss value
            loss = sum_nodes(losses) / len(losses)
            print(f"Epoch({k+1}/{epochs}) | Loss: {loss._vl}")

    def predict(self, inputs, true_ys):
        pred_ys = []
        for input in inputs:
            logits = self.model(input)
            sum = sum_nodes([x.exp() for x in logits])
            probs = [x.exp() / sum for x in logits]
            pred_ys.append(np.argmax(probs))
        print(f"Pred: {pred_ys}")
        print(f"True: {true_ys}")
