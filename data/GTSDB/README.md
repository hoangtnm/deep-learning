# The German Traffic Sign Detection Benchmark

**The German Traffic Sign Detection Benchmark** is a single-image detection assessment for researchers with interest in the field of computer vision, pattern recognition and image-based driver assistance. It is supposed to be introduced on the [IEEE International Joint Conference](http://www.ijcnn2013.org/) on Neural Networks 2013. It features ...

- a single-image detection problem
- 900 images (devided in 600 training images and 300 evaluation images)
- division into three categories system that suit the properties of various detection approaches with different properties
- an online evaluation system with immediate analysis and ranking of the submitted results

## Competion Test Dataset

If you want to participate in our competition, please download the [test dataset](http://benchmark.ini.rub.de/Dataset_GTSDB/TestIJCNN2013.zip) (530 MB) and process the images by your traffic sign detector. For details on the submission procedure, please refer to section "Submission format and regulations".

## Download

Feel free to download the [full data set](http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip) (1.6 GB) and use it for your purposes. The download package comprises

- the 900 training images (1360 x 800 pixels) in PPM format
- the image sections containing only the traffic signs
- a file in CSV format containing ground truth
- a ReadMe.txt explainig some details

## Image format

- The images contain zero to six traffic signs. However, even if there is a traffic sign located in the image it may not belong to the competition relevant categories (prohibitive, danger, mandatory).
- Images are stored in PPM format
- The sizes of the traffic signs in the images vary from 16x16 to 128x128
- Traffic signs may appear in every perspective and under every lighting condition

## Annotation format

Annotations are provided in CSV files. Fields are seperated by a semicolon (;). They contain the following information:

- Filename: Filename of the image the annotations apply for
- Traffic sign's region of interest (ROI) in the image
  - leftmost image column of the ROI
  - upmost image row of the ROI
  - rightmost image column of the ROI
  - downmost image row of the ROI
- ID providing the traffic sign's class
