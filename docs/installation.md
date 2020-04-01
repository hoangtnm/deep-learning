# Installation

## Prerequisites

- OS: Ubuntu 18.04 LTS
- Programming language: Python 3

In case you want to know how to build and setup a machine ready for deep learning, please read [this document](https://github.com/hoangtnm/docs/blob/master/Machine_Setup.md).

## Anaconda installation

Anaconda® is a package manager, an environment manager, a Python/R data science distribution, and a collection of [over 7,500+ open-source packages](https://docs.anaconda.com/anaconda/packages/pkg-docs/). Anaconda is free and easy to install, and it offers [free community support](https://groups.google.com/a/anaconda.com/forum/?fromgroups#!forum/anaconda).

Get the [Anaconda Cheat Sheet](https://docs.anaconda.com/_downloads/9ee215ff15fde24bf01791d719084950/Anaconda-Starter-Guide.pdf) and then [download Anaconda](https://www.anaconda.com/downloads).

After installing Anaconda, it is recommended to `add some channels to the top of the channel list`, making it the highest priority, in order to get libraries which are `highly-optimized` or not available by default:

```sh
conda config --add channels pytorch
# EXPERIMENT: Installing the Intel Distribution for Python
# conda config --append channels intel
# conda create -n idp intelpython3_full python=3
```

## Libraries installation

```sh
conda install -f requirements_conda.txt
conda install -r requirements_pip.txt
```

## Add Libraries to PYTHONPATH

When running locally, the deep-learning directory should be appended to `PYTHONPATH`. This can be done by running the following from deep-learning:

```sh
# From deep-learning/
export PYTHONPATH=$PYTHONPATH:`pwd`
```

Note: This command needs to run from every new terminal you start. If you wish to avoid running this manually, you can add it as a new line to the end of your ~/.bashrc file, replacing `pwd` with the absolute path of deep-learning on your system.
