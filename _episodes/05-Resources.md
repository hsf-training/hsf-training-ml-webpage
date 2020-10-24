---
title: "Resources"
teaching: 10
exercises: 0
questions:
- "Where should I go if I want to get better at Python?"
- "What are the machine learning libraries in Python?"
objectives:
- "Provide links to tutorials and textbooks that will help you get better at Python."
- "Provide links to machine learning library documentation."
keypoints:
- "NumPy and pandas are the main libraries for scientific computing."
- "scikit-learn and PyTorch are two good options for machine learning in Python."
---

<iframe width="427" height="251" src="https://www.youtube.com/embed?v=AGCps9YmWK8&list=PLKZ9c4ONm-VmHsMKImIDEMsZI1Vp0UY-Z&index=4&ab_channel=HEPSoftwareFoundation" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# Proficiency in Python

If you are unfamiliar with Python, the following tutorials will be useful:

* [Python novice inflammation](https://swcarpentry.github.io/python-novice-inflammation/)
* [Python novice gapfinder](http://swcarpentry.github.io/python-novice-gapminder/)

For non-trivial machine learning tasks that occur in research, one needs to be proficient in the programming libraries discussed in the tutorial here. There are two main python libraries for scientific computing:

1. **NumPy**: the go-to numerical library in Python. See the [documentation](https://numpy.org/). NumPy's main purpose is the manipulation of multi-dimensional arrays: this includes both

   * *slicing*: taking "chunks" out of arrays. Slicing in Python means taking elements from one given index to another given index. For 1 dimensional arrays this reduces to selecting intervals, but these operations can become quite advanced for multidimesional arrays.
   * *functional operations*: applying a function to an entire array. This code is highly optimized: there is often a myth that Python is slower than languages like C++; while this may be true for things like for-loops, it is not true if you use NumPy properly.

2. **pandas**: pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language. See the [documentation](https://pandas.pydata.org/). The most important datatype in the pandas library is the *DataFrame*: a "spreadsheet-type object" with row and column names. It is preferred to use pandas DataFrames rather than NumPy arrays for managing data sets.

If you are unfamiliar with these packages, I would recommend reading the introduction to the documentation pages for [NumPy slicing](https://numpy.org/doc/stable/reference/arrays.indexing.html), [NumPy operations](https://scipy-lectures.org/intro/numpy/operations.html) and [pandas DataFrames](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html) or sitting down with [this textbook](https://www.amazon.ca/Python-Data-Analysis-Wrangling-IPython-ebook/dp/B075X4LT6K/ref=sr_1_1?crid=WLIHOCVH891S&dchild=1&keywords=python+for+data+analysis%2C+2nd+edition&qid=1593460237&sprefix=python+for+data+%2Caps%2C196&sr=8-1) if you have it available and reading/coding along with chapters 4 and 5. In a few hours, you should have a good idea of how these packages work.

# Machine Learning Libraries in Python

There are many machine libraries in Python, but the two discussed in this tutorial are scikit-learn and TensorFlow.

1. **scikit-learn**: features various classification, regression and clustering algorithms and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy. See [the documentation](https://scikit-learn.org/stable/).
2. **PyTorch**: PyTorch is an end-to-end open-source platform for machine learning. It is used for building and deploying machine learning models. See [the documentation](https://www.tensorflow.org/).

> ## What about GPUs?
> **scikit-learn** doesn't have GPU support, therefore should only be used for training simple neural networks.
> **PyTorch** does have GPU support, therefore can be used train complicated neural network model that require a lot of GPU power.
{: .callout}

Note that the four Python programming packages discussed so far are interoperable: in particular, datatypes from NumPy and pandas are often used in packages like scikit-learn and PyTorch.
