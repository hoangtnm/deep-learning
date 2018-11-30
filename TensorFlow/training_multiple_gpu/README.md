# Training a Model Using Multiple GPU Cards

In a workstation with multiple GPU cards, each GPU will have similar speed and contain enough memory to run an entire model.

Thus, we opt to design our training system in the following manner:

- Place an individual model replica on each GPU
- Update model parameters synchronously by waiting for all GPUs to finish processing a batch of data

![Multi-GPU architecture](https://www.tensorflow.org/images/Parallelism.png)

Note:

- Each GPU computes a unique batch of data (dividing up a larger batch of data across the GPUs)
- All GPUs share the model parameters (store and update all the parameters on the CPU)
- A fresh set of model parameters is transferred to the GPU when a new batch of data is processed by all GPUs
- The GPUs are synchronized in operation. All gradients are accumulated from the GPUs and averaged. The model parameters are updated with the gradients averaged across all model replicas

## Placing Variables and Operations on Devices

The first required abstraction is a function for computing inference and gradients for a single model replica ("tower"):

- A unique name for all operations within a tower (*tf.name_scope*). For instance, all operations in the first tower are prepended with `tower_0`, e.g. `tower_0/conv1/Conv2D`
- A preferred hardware device to run the operation within a tower (*tf.device*). For instance, all operations in the first tower reside within device(`'/device:GPU:0'`)

## Launching and Training the Model on Multiple GPU cards

If you have several GPU cards installed on your machine you can use them to train the model faster with the `cifar10_multi_gpu_train.py` script. This version of the training script parallelizes the model across multiple GPU cards.

```bash
python cifar10_multi_gpu_train.py --num_gpus=2
```