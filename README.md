# Deep Learning Research

The goal of this repository is to provide comprehensive tutorials and experiments for TensorFlow and PyTorch while maintaining the simplicity of the code.

<!-- <div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_social.png">
</div> -->

<!-- [TensorFlow](https://www.tensorflow.org/) is an end-to-end open source platform for machine learning. It has a comprehensive, flexible ecosystem of [tools](https://www.tensorflow.org/resources/tools), [libraries](https://www.tensorflow.org/resources/libraries-extensions), and [community](https://www.tensorflow.org/community) resources that lets researchers push the state-of-the-art in ML and developers easily build and deploy ML-powered applications. -->

<!-- <div align="center">
  <img src="https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png">
</div> -->

<!-- [PyTorch](https://pytorch.org/) is an open source machine learning framework that accelerates the path from research prototyping to production deployment. -->

<!-- ## Installation

```sh
conda install tensorflow-gpu
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
``` -->

## Contents

## Tutorials

- Machine Learning Crash Course by Google
- Machine Learning Course by Stanford University
- Natural Language Processing in TensorFlow by deeplearning.ai
- Sequence Models Course by deeplearning.ai
- PyImageSearch code snippets

## Tips & Tricks

### Tensor Core Performance

#### Dimension Size

- `GEMMs = "generalized (dense) matrix-matrix multiplies"`:\
  For A x B where A has size (M, K) and B has size (K, N):
  > N, M, K should be multiplies of 8.
- `GEMMS in fully connected layers:`\
  Batch size, input features, output features shoule be multiplies of 8.
- `GEMMs in RNNs:`\
  Batch size, hidden size, embedding size, dictionary size shoule be multiplies of 8.

Libraries (cuDNN, cuBLAS) are optimized for Tensor Cores.

#### cuDNN Algorithms

If your data/layer sizes are constant each iteration, try

```python
import torch
torch.backend.cudnn.benchmark = True
...
```

This enables PyTorch's autotuner.

The first iteration, it will test different cuDNN algorithms for each new convolutions size it sees, and cache the fastest choice to use in later iterations. [Details](https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936)

### Named Entity Recognition

The CoNLL-2003 shared task data files contain four columns separated by a single space. Each word has been put on a separate line and there is an empty line after each sentence.

In a pre-processing step only the two relevant columns (token and outer span NER annotation) are extracted:

```sh
export MAX_LENGTH=128
export BERT_MODEL=bert-base-multilingual-cased

cat train.txt | grep -v "^#" | cut -f 1,4 | tr '\t' ' ' > train.txt.tmp
python utils/ner_preprocess.py train.txt.tmp $BERT_MODEL $MAX_LENGTH > train.txt
```
