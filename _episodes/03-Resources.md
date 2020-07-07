---
title: "Resources"
teaching: 10
exercises: 0
questions:
- "Where should I go if I want to get better at python?"
- "What are the machine learning libraries in python?"
- "Where should I go if I want to get better at machine learning?"
objectives:
- "Provide links to textbooks that will help you get better at python."
- "Give links to machine learning library documentation."
- "Provide links to machine learning textbooks in python."
keypoints:
- "Textbook provided for learning NumPy and pandas."
- "Sci-kit learn and Tensorflow are two good options for machine learning in python."
- "Textbook provided for learning machine learning libraries."
---

# Proficiency in python

If you are unfamiliar with python, the following tutorials will be useful:

* [python novice inflammation](https://swcarpentry.github.io/python-novice-inflammation/)
* [python novice gapfinder](http://swcarpentry.github.io/python-novice-gapminder/)

For non-trivial machine learning tasks that occur in research, one needs to be proficient in the programming libraries discussed in the tutorial here. There are three main python libraries for scientific computing:

1. **numpy**: the go-to numerical library in python. See the [documentation](https://numpy.org/). numpy's main purpose is the manipulation of multi-dimensional arrays: this includes both

* *slicing*: taking "chunks" out of arrays. Slicing in python means taking elements from one given index to another given index. For 1 dimensional arrays this reduces selecting intervals, but these operations can become quite advanced for multidimesional arrays.
* *functional operatorions*: applying a function to an entire array . This code is highly optimized: there is often a myth that python is slower than languages like C++; while this may be true for things like for-loops, it is not true if you use numpy properly.

2. **pandas**: pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language. The most important datatype in the pandas library is the *DataFrame*: a "spreadsheet-type object" with row and column names. It is preferred to use pandas DataFrames than numpy arrays for managing data sets.

If you are unfamiliar with these packages, I would recommend sitting down with [this textbook](https://www.amazon.ca/Python-Data-Analysis-Wrangling-IPython-ebook/dp/B075X4LT6K/ref=sr_1_1?crid=WLIHOCVH891S&dchild=1&keywords=python+for+data+analysis%2C+2nd+edition&qid=1593460237&sprefix=python+for+data+%2Caps%2C196&sr=8-1) and reading/coding along with chapters 4 and 5. In a few hours, you should have a good idea of how these packages work.

# Machine Learning Libraries in python

There are many machine libraries in python, but the two discussed in this tutorial are sci-kit learn and tensorflow.

1. **sci-kit learn**: features various classification, regression and clustering algorithms and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy. This library does **not** have gpu support: as such, it should not be used for training neural networks. See [the documentation](https://scikit-learn.org/stable/)
2. **tensorflow**: TensorFlow is an end-to-end open source platform for machine learning. It is used for building and deploying machine learning models and **does** have gpu support. See [the documentation](https://www.tensorflow.org/). Tensorflow is used to train complicated neural network models that require a lot of gpu power.

Note that the four python programming packages discussed so far are interoperable: in particular, datatypes from numpy and pandas are often used in packages like sci-kit learn and tensorflow.

# Code Example

Here we will import all the required libraries for the rest of the tutorial. All sci-kit learn functions will be imported later on when they are required.

~~~
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow import keras
~~~
{: .language-python}

You can check the version of these packages by checking the `__version__` attribute.

~~~
np.__version__
~~~
{: .language-python}




