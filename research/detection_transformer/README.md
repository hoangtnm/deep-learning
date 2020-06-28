# End-to-End Object Detection with Transformers <!-- omit in toc -->

TensorFlow version of **DETR** (**DE**tection **TR**ansformer) based on the [original PyTorch implementation](https://github.com/facebookresearch/detr).

## Contents <!-- omit in toc -->

- [Abstract](#abstract)
- [References](#references)

## Abstract

DETR stands for _DEtection TRansformer_, which is a new method that views object detection as a direct set prediction problem.
It streamlines the detection pipeline, effectively removing the need for many hand-designed components like a non-maximum suppression procedure or anchor generation that explicitly encode our prior knowledge about the task.
The main ingredients of DETR, are a set-based global loss that forces unique predictions via bi-partite matching, and a transformer encoder-decoder architecture. [1]

DETR relies on a simple yet powerful mechanism called attention, which enables it to selectively focus on certain parts of their input or reason about the relations of the objects and the global image context to directly output the final set of predictions in parallel.

> DETR is the first object detection framework to successfully integrate Transformers as a central building block in the detection pipeline.

In terms of accuracy and performance, DETR matches the well-established and highly-optimized Faster RCNN baseline on COCO dataset, while also greatly simplifying and streamlining the architecture.

<p align="center">
    <img src="https://scontent.fsgn2-6.fna.fbcdn.net/v/t39.2365-6/99436670_2434253423531845_6527599147384569856_n.jpg?_nc_cat=110&_nc_sid=ad8a9d&_nc_ohc=Fw-JIv74v3EAX8ou2vJ&_nc_ht=scontent.fsgn2-6.fna&oh=185c98ba902db500dc36f75897a66354&oe=5F1CD744">
</p>

## References

[1] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and S. Zagoruyko,“End-to-end object detection with transformers,” _arXiv preprint arXiv:2005.12872_,2020.
