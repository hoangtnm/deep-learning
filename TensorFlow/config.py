#! /usr/bin/python

#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import tensorflow as tf

"""Supported devices
/cpu:0: The CPU of your machine.
/device:GPU:0: The GPU of your machine, if you have one.
/device:GPU:1: The second GPU of your machine, etc.
"""

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1,
							allow_growth=True,
							visible_device_list = "0")
config=tf.ConfigProto(
				allow_soft_placement=False,
				log_device_placement=True,
				gpu_options=gpu_options
				)

"""Example of output
session = tf.Session(config=config, ...)
Device mapping:
/job:localhost/replica:0/task:0/device:GPU:0 -> device: 0, name: Tesla K40c, pci bus
id: 0000:05:00.0
b: /job:localhost/replica:0/task:0/device:GPU:0
a: /job:localhost/replica:0/task:0/device:GPU:0
MatMul: /job:localhost/replica:0/task:0/device:GPU:0
[[ 22.  28.]
 [ 49.  64.]]
"""

"""Advanced settings
// list inside each element. For example,
//   visible_device_list = "1,0"
//   virtual_devices { memory_limit: 1GB memory_limit: 2GB }
//   virtual_devices {}
// will create three virtual devices as:
//   /device:GPU:0 -> visible GPU 1 with 1GB memory
//   /device:GPU:1 -> visible GPU 1 with 2GB memory
//   /device:GPU:2 -> visible GPU 0 with all available memory
"""
