# Action Recognition <!-- omit in toc -->

## Contents <!-- omit in toc -->

- [Model Architectures](#model-architectures)
  - [C3D](#c3d)
- [Dataset Preparation](#dataset-preparation)
  - [UCF101 Dataset](#ucf101-dataset)
- [Training](#training)
- [References](#references)

## Model Architectures

### C3D

C3D is a modified version of BVLC caffe to support 3D convolution and pooling. The main supporting features include:

- Training or fine-tuning 3D ConvNets.
- Extracting video features with pre-trained C3D models.

For more information about C3D, please refer to the [C3D project website](http://vlg.cs.dartmouth.edu/c3d).

## Dataset Preparation

### UCF101 Dataset

Please follow [UCF101 instructions](../../datasets/ucf101) for more information.

## Training

```python
python train.py --input_dir <path>/<to>/<TFRecord folder>
```

## References

- <a href='https://arxiv.org/pdf/1412.0767.pdf'>Learning Spatiotemporal Features with 3D Convolutional Networks</a><br>
