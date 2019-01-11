# Machine Learning & Deep Learning Research

The goal of this repository is to provide comprehensive tutorials and experiments for TensorFlow while maintaining the simplicity of the code.

<p align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png"><br><br>
</p>

-----------------

**TensorFlow** is an open source software library for numerical computation
using data flow graphs. The graph nodes represent mathematical operations, while
the graph edges represent the multidimensional data arrays (tensors) that flow
between them. This flexible architecture enables you to deploy computation to
one or more CPUs or GPUs in a desktop, server, or mobile device without
rewriting code. TensorFlow also includes
[TensorBoard](https://github.com/tensorflow/tensorboard), a data visualization
toolkit.

TensorFlow was originally developed by researchers and engineers
working on the Google Brain team within Google's Machine Intelligence Research
organization for the purposes of conducting machine learning and deep neural
networks research.  The system is general enough to be applicable in a wide
variety of other domains, as well.

## Installation

To install the current release for CPU-only:

```
pip install tensorflow
```

Use the GPU package for CUDA-enabled GPU cards:

```
pip install tensorflow-gpu
```

*See [Installing TensorFlow](https://www.tensorflow.org/install) for detailed
instructions, and how to build from source.*

#### *Try your first TensorFlow program*

```shell
$ python
```
```python
>>> import tensorflow as tf
>>> tf.enable_eager_execution()
>>> tf.add(1, 2)
3
>>> hello = tf.constant('Hello, TensorFlow!')
>>> hello.numpy()
'Hello, TensorFlow!'
```
Learn more examples about how to do specific tasks in TensorFlow at the [tutorials page of tensorflow.org](https://www.tensorflow.org/tutorials/).

## Contact Information

If you have any questions or pull requests, please feel free to contact me. You can communicate with me by sending e-mail to minhhoangtrannhat@gmail.com.
