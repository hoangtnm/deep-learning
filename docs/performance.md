# Performance guidelines and practical recommendatons

## Table of contents

Mixed precision:

  * <a href='#11-debugging-mixed-precision'>Debugging mixed precision</a><br>

Getting the most from Tensor Cores:

  * <a href='#21-satisfy-tensor-core-shape-constraints'>Satisfy Tensor Core shape constraints</a><br>
  * <a href='#22-increase-arithmetic-intensity'>Increase arithmetic intensity</a><br>
  * <a href='#23-decrease-fraction-of-work-in-non-tensor-core-ops'>Decrease fraction of work in non-Tensor Core ops</a><br>

## 1. Mixed precision:

### 1.1. Debugging mixed precision

- The `unreasonable effectiveness of gradient descent`
  - Bugs in code for mixed precision steps open manifest as slightly worsetraining accuracy
- Common mistakes:
  - `Gradients not unscaled correctly` before weight update (Adam will try to handle this!)
  - Gradient clipping or regularization improperly using scaled gradients
  - Incorrectly synchronizing master weight updates across multiple GPUs
  - Not running `loss function in FP32`

## 2. Getting the most from Tensor Cores

Three levels of optimization to best use Tensor Cores:

- Satisfy Tensor Core shape constraints
- Increase arithmetic intensity
- Decrease fraction of work in non-Tensor Core ops

### 2.1. Satisfy Tensor Core shape constraints

- GEMMs = `generalized (dense) matrix-matrix multiplies`\
  All three dimensions (M, N, K) should be multiples of 8

- GEMMs in `fully connected layers`:\
  Batch size, input and output features should be multiples of 8

- GEMMs in `RNNs`:\
  Batch size, hidden size, embedding size, and dictionary size should bemultiples of 8

- `Convolution`:\
  Number of `channels` (input, output) should be multiples of 8

- In practice:

  - Choose `minibatch` a multiples of 8
  - Choose `layer dimensions` to be multiples of 8
  - For classification, pad `vocabulary` to a multiples of 8
  - For sequence, pad `sequence length` to a multiples of 8

**Enabling PyTorch’s autotuner**:

```python
import torch
torch.backends.cudnn.benchmark = True
...
```

The first iteration, it will test different cuDNN algorithms for each new convolution size it sees, and cache the fastest choice to use in later iteration.
[Details](https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936)

### 2.2. Increase arithmetic intensity

- Increase arithmetic intensity in model `implementation`:
  - Concatenate weights and gate activations in recurrent cells
  - Concatenate activations across time in sequence models

- Increase arithmetic intensity in model `architecture`:
  - Prefer dense math (vanilla convolutions vs. depth separable convolutions)
  - Prefer wider layers - often little speed cost
  - Of course, `always prefer accuracy first!`

### 2.3. Decrease fraction of work in non-Tensor Core ops

- Cutting-edge work on speeding up non-Tensor Core ops automatically with compiler tools:
  - TensorFlow: XLA
  - PyTorch JIT
