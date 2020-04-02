# Deep Learning Research

The goal of this repository is to provide comprehensive tutorials and experiments for TensorFlow 2 and PyTorch while maintaining the simplicity of the code.

## Prerequisites

It is assumed that you have some Python programming experience and that you are familiar with Python’s main scientific libraries—in particular, [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), and [Matplotlib](https://matplotlib.org/).

Also, if you care about what’s under the hood, you should have a reasonable understanding of college-level math as well (calculus, linear algebra, probabilities, and statistics).

If you don’t know Python yet, http://learnpython.org/ is a great place to start. The official tutorial on Python.org is also quite good.

## Table of contents

Setup:

- <a href='docs/installation.md'>Installation</a><br>

<!-- Research: -->

Tutorials:

- Machine Learning Crash Course by Google
- Machine Learning Course by Stanford University
- Natural Language Processing in TensorFlow by deeplearning.ai
- Sequence Models Course by deeplearning.ai
- PyImageSearch code snippets
- <a href='tutorials/recap/README.md'>Deep Learning Recap</a><br>

Guidelines:

- <a href='docs/performance.md'>Performance guidelines and practical recommendatons</a><br>
- <a href='docs/ner.md'>Entity Recognition pre-processing</a><br>

Framework introduction:

- <a href='#tensorflow'>TensorFlow</a><br>
- <a href='#pytorch'>PyTorch</a><br>

## Framework introduction

### TensorFlow

[TensorFlow](https://www.tensorflow.org/) is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of [tools](https://www.tensorflow.org/resources/tools), [libraries](https://www.tensorflow.org/resources/libraries-extensions), and [community](https://www.tensorflow.org/community) resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML-powered applications.

TensorFlow was originally developed by researchers and engineers working on the Google Brain team within Google's Machine Intelligence Research organization to conduct machine learning and deep neural networks research. The system is general enough to be applicable in a wide variety of other domains, as well.

### PyTorch

![PyTorch Logo](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png)

---

#### Overview

PyTorch is a Python package that provides two high-level features:

- Tensor computation (like NumPy) with strong GPU acceleration.

- Deep neural networks built on a tape-based autograd system.

#### Key Features & Capabilities

- Hybrid Front-End: A new hybrid front-end seamlessly transitions between eager mode and graph mode to provide both flexibility and speed.
- Distributed Training: Scalable distributed training and performance optimization in research and production is enabled by the torch.distributed backend.
- Python-First: Deep integration into Python allows popular libraries and packages to be used for easily writing neural network layers in Python.
- Tools & Libraries: A rich ecosystem of tools and libraries extends PyTorch and supports development in computer vision, NLP and more.

- [More about PyTorch](#more-about-pytorch)
- [Getting Started](#getting-started)

#### More About PyTorch

At a granular level, PyTorch is a library that consists of the following components:

| Component                 | Description                                                                                                                             |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **torch**                 | a Tensor library like NumPy, with strong GPU support                                                                                    |
| **torch.autograd**        | a tape-based automatic differentiation library that supports all differentiable Tensor operations in torch                              |
| **torch.nn**              | a neural networks library deeply integrated with autograd designed for maximum flexibility                                              |
| **torch.multiprocessing** | Python multiprocessing, but with magical memory sharing of torch Tensors across processes. Useful for data loading and Hogwild training |
| **torch.utils**           | DataLoader, Trainer and other utility functions for convenience                                                                         |

Usually one uses PyTorch either as:

- a replacement for NumPy to use the power of GPUs.
- a deep learning research platform that provides maximum flexibility and speed.

Elaborating further:

##### A GPU-Ready Tensor Library

If you use NumPy, then you have used Tensors (a.k.a ndarray).

![Tensor illustration](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/tensor_illustration.png)

PyTorch provides Tensors that can live either on the CPU or the GPU, and accelerates the
computation by a huge amount.

We provide a wide variety of tensor routines to accelerate and fit your scientific computation needs
such as slicing, indexing, math operations, linear algebra, reductions.
And they are fast!

##### Dynamic Neural Networks: Tape-Based Autograd

PyTorch has a unique way of building neural networks: using and replaying a tape recorder.

Most frameworks such as TensorFlow, Theano, Caffe and CNTK have a static view of the world.
One has to build a neural network, and reuse the same structure again and again.
Changing the way the network behaves means that one has to start from scratch.

With PyTorch, we use a technique called reverse-mode auto-differentiation, which allows you to
change the way your network behaves arbitrarily with zero lag or overhead. Our inspiration comes
from several research papers on this topic, as well as current and past work such as
[torch-autograd](https://github.com/twitter/torch-autograd),
[autograd](https://github.com/HIPS/autograd),
[Chainer](https://chainer.org), etc.

While this technique is not unique to PyTorch, it's one of the fastest implementations of it to date.
You get the best of speed and flexibility for your crazy research.

![Dynamic graph](https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/dynamic_graph.gif)

##### Python First

PyTorch is not a Python binding into a monolithic C++ framework.
It is built to be deeply integrated into Python.
You can use it naturally like you would use [NumPy](http://www.numpy.org/) / [SciPy](https://www.scipy.org/) / [scikit-learn](http://scikit-learn.org) etc.
You can write your new neural network layers in Python itself, using your favorite libraries
and use packages such as Cython and Numba.
Our goal is to not reinvent the wheel where appropriate.

##### Imperative Experiences

PyTorch is designed to be intuitive, linear in thought and easy to use.
When you execute a line of code, it gets executed. There isn't an asynchronous view of the world.
When you drop into a debugger, or receive error messages and stack traces, understanding them is straightforward.
The stack trace points to exactly where your code was defined.
We hope you never spend hours debugging your code because of bad stack traces or asynchronous and opaque execution engines.

##### Fast and Lean

PyTorch has minimal framework overhead. We integrate acceleration libraries
such as [Intel MKL](https://software.intel.com/mkl) and NVIDIA (cuDNN, NCCL) to maximize speed.
At the core, its CPU and GPU Tensor and neural network backends
(TH, THC, THNN, THCUNN) are mature and have been tested for years.

Hence, PyTorch is quite fast – whether you run small or large neural networks.

The memory usage in PyTorch is extremely efficient compared to Torch or some of the alternatives.
We've written custom memory allocators for the GPU to make sure that
your deep learning models are maximally memory efficient.
This enables you to train bigger deep learning models than before.

##### Extensions Without Pain

Writing new neural network modules, or interfacing with PyTorch's Tensor API was designed to be straightforward
and with minimal abstractions.

You can write new neural network layers in Python using the torch API
[or your favorite NumPy-based libraries such as SciPy](https://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html).

If you want to write your layers in C/C++, we provide a convenient extension API that is efficient and with minimal boilerplate.
There is no wrapper code that needs to be written. You can see [a tutorial here](https://pytorch.org/tutorials/advanced/cpp_extension.html) and [an example here](https://github.com/pytorch/extension-cpp).

#### Getting Started

Three pointers to get you started:

- [Tutorials: get you started with understanding and using PyTorch](https://pytorch.org/tutorials/)
- [Examples: easy to understand pytorch code across all domains](https://github.com/pytorch/examples)
- [The API Reference](https://pytorch.org/docs/)

<!-- ## Tips & Tricks -->
