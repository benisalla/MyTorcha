# MyTorcha

**MyTorcha** is a small framework (a baby framework that hopes to grow into something bigger, Inshallah). Currently, it's very simple and includes a neuron, layer, and a mechanism to stack layers according to your needs. You can find more details about its components below.

The framework is implemented in Python for now, but I plan to release the C++ version soon, Inshallah. The core concept behind building neural networks with this framework is rooted in graph theory, particularly a directed acyclic graph (DAG). This graph contains all the relevant information about weights, biases, derivatives, and activation functions, which will be explained in the following sections.

<div align="center">
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/288189698-874ae81f-1951-4698-af3a-ee28468703ac.gif" width="800" height="400" alt="Graph Visualization"/>
</div>

Is this the end? No! As long as I’m breathing, none of my projects will stop (hopefully). I am currently working on ConvNets and Transformers based on the same graph-theoretic approach (almost done, just fixing some bugs).

<div align="center">
  <h5>MyTorcha</h5>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/288193776-a2eb7531-5305-4c5b-98d5-875bfb619a08.png" width="400" height="400" alt="MyTorcha Logo"/>
</div>

---

## Table of Contents

- [Introduction](#introduction)
- [Motivation](#motivation)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Future Roadmap](#future-roadmap)
- [Contributing](#contributing)
- [Contact](#contact)

---

## Introduction

**MyTorcha** is inspired by my deep curiosity about how neural networks operate under the hood. While mainstream frameworks like PyTorch and TensorFlow offer powerful, production-ready tools, they often abstract away much of the internal mechanics. **MyTorcha** aims to provide an educational tool for anyone looking to gain a better understanding of neural networks from the ground up.

The framework leverages graph theory, particularly **directed acyclic graphs (DAGs)**, to model computational dependencies between the different components of a neural network, such as weights, biases, and activation functions.

---

## Motivation

I believe that complexity is one of the primary challenges of modern software frameworks. Using pre-built tools like PyTorch and TensorFlow is efficient but often leaves me feeling disconnected from the underlying processes. That’s why I embarked on this journey—to create something that allows for greater transparency and control while fostering a deeper understanding of neural networks.

This project is driven by my personal belief:

> **"The Deal is in the Details"**

---

## Features

- **Custom Neural Network Components**: Build networks using custom-defined neurons and layers.
- **Graph-based Computation**: Leverages DAGs to model computation flow.
- **Lightweight**: A simple and easy-to-use framework for learning and experimentation.
- **Modular**: Extendable to include new types of layers and activation functions.
- **Educational Focus**: Learn the intricacies of training, forward passes, and backpropagation.

---

## Model Architecture

At its core, **MyTorcha** lets users build fully connected networks (and other architectures in progress) with a straightforward, intuitive API. 

- **Layers**: Configure fully connected or custom layers.
- **Activation Functions**: Includes support for common activation functions like **ReLU**.
- **Optimizer**: RMSProp is implemented for efficient training.

<div align="center">
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/273406292-81226d87-8166-4f6d-b913-baa419f794ff.png" width="600" height="300" alt="Loss Function Visualization"/>
</div>

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/benisalla/MyTorcha.git
cd MyTorcha
pip install -r requirements.txt
```

---

## Usage

Here’s a simple example of creating a neural network using **MyTorcha**:

```python
from MyTorcha.Layer import Layer
from MyTorcha.Neuron import Neuron
from Models.SimpleFC import SimpleFC
from NN.Trainer import Trainer

# Initialize your model
model = SimpleFC(input_size=784, hidden_layers=[128, 64], output_size=10, act_fun='relu')

# Set up the trainer
trainer = Trainer(model=model, lr=0.01, batch_size=32, loss="cross_entropy", optimizer="rms_prop")

# Load data (e.g., MNIST dataset)
X_train, Y_train = load_data()

# Train your model
trainer.fit(X_train, Y_train, epochs=10)

# Predict
predictions = trainer.predict(X_test)
```

---

## Training

Training in **MyTorcha** is straightforward. The `Trainer` class handles both forward and backward passes and optimizes network parameters using different techniques like RMSProp.

You can visualize the loss and monitor the training process easily using built-in logging functionality.

---

## Future Roadmap

Planned Features:
- **Convolutional Neural Networks (CNNs)** support.
- **Transformer models** for NLP and vision tasks.
- Improved optimization algorithms like Adam and SGD with momentum.
- **Graph visualization** tools for better understanding the computational graph.
- A **C++ backend** for faster execution.

---

## Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request for any improvements or new features. 

Steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push the branch (`git push origin feature-branch`).
5. Open a pull request.

---

## Contact

For any inquiries or suggestions, feel free to reach out:

- **Email**: [ismailbenalla52@gmail.com](mailto:ismailbenalla52@gmail.com)
- **Twitter**: [@ismail_ben_alla](https://twitter.com/ismail_ben_alla)
- **LinkedIn**: [Ismail Ben Alla](https://linkedin.com/in/ismail-ben-alla-7144b5221/)
- **Instagram**: [@ismail_ben_alla](https://instagram.com/ismail_ben_alla)

Let’s connect and explore the world of AI together! 🤖🌟

<div align="center">
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/288196444-aae5a2b5-6623-4f1e-9149-be45342093c3.gif" width="500" height="300" alt="Neural Net Humor"/>
</div>
