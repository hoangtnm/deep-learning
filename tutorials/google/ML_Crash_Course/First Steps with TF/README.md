# Common hyperparameters

- **steps**: the total number of training iterations. One step calculates the loss from one batch and uses that value to modify the model's weights once.
- **batch size**: the number of examples (chosen at random) for a single step. For example, the batch size for SGD is 1.

```
                                  The following formula applies:
                      total number of trained examples = batch size * steps
```

Quick Introduction to pandas: pandas is an important library for data analysis and modeling, and is widely used in TensorFlow coding. This tutorial provides all the pandas information you need for this course.

First Steps with TensorFlow: This exercise explores linear regression.

Synthetic Features and Outliers: This exercise explores synthetic features and the effect of input outliers.  


# A convenience variable in Machine Learning Crash Course exercises

**periods,** which controls the granularity of reporting.

For example, if periods is set to 7 and steps is set to 70, then the exercise will output the loss value every 10 steps (or 7 times)

Note that modifying periods does not alter what your model learns
