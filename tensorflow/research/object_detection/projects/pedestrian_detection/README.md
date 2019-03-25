# Pedestrian-Detection

Pedestrian Detection using the TensorFlow Object Detection API 

<p align="center">
  <img src="markdown_images/output.gif">
</p>

This project provides complementary material to this [blog post](https://medium.com/nanonets/how-to-automate-surveillance-easily-with-deep-learning-4eb4fa0cd68d), which compares the performance of four object detectors for a pedestrian detection task. It also introduces a feature to use multiple GPUs in parallel for inference using the multiprocessing package. The count accuracy and FPS for different models (using 1,2,4 or 8 GPUs in parallel) were calculated and plotted.

## Dataset

The [TownCentre](http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/project.html#datasets) dataset is used for training our pedestrian detector. You can use the following commands to download the dataset. This automatically extracts the frames from the video, and creates XML files from the csv groundtruth. The image dimensions are downscaled by a factor of 2 to reduce processing overhead.

```shell
wget http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/Datasets/TownCentreXVID.avi
wget http://www.robots.ox.ac.uk/ActiveVision/Research/Projects/2009bbenfold_headpose/Datasets/TownCentre-groundtruth.top
python extract_towncentre.py
python extract_GT.py
```

## Setup

### 1. TensorFlow Object Detection API

Refer to the instructions in this [blog post](https://medium.com/nanonets/how-to-automate-surveillance-easily-with-deep-learning-4eb4fa0cd68d).

### 2. Instructions

```python

```

## Results

### FPS vs GPUs
<p align="center">
  <img src="markdown_images/fps.png" alt="FPS vs GPUs"></img>
</p>

For more stats, refer to the [blog post](https://medium.com/nanonets/how-to-automate-surveillance-easily-with-deep-learning-4eb4fa0cd68d).
The performance of each model (on the test set) was compiled into a video, which you can see [here](https://www.youtube.com/watch?v=0hWW6FVcFAo).
