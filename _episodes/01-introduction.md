---
title: "Introduction"
teaching: 0
exercises: 0
questions:
- "What is machine learning?"
- "What role does machine learning have in particle physics?"
- "What should I do if I want to get good at machine learning?"
objectives:
- "Discuss the general learning task in machine learning."
- "Provide examples of machine learning in particle physics."
- "Give resources to people who want to become proficient in machine learning."
keypoints:
- "In general, machine learning is about designing a function $$f$$ such that $$y=f(x)$$ fits a dataset $$(x_i,y_i)$$. The domain and range of $$f$$ aren't necessarily real numbers: in fact, they are often much more complicated."
- "Machine learning has many applications in particle physics."
- "If you want to become proficient in machine learning, you need to practice."
---

# What is Machine Learning?

General definition

<ul>
[Machine Learning is the] field of study that gives computers the ability to learn without being explicitly programmed
  <ul>
    -Arthur Samuel, 1959
  </ul>
</ul>

In a **traditional approach** to solving problems, one would study a problem, write rules (i.e. laws of physics) to solve that problem, analyze errors, then modify the rules. A **machine learning approach** automates this process: *a machine learning model modifies its rules based on the errors it measures*. This abstract statement becomes more conceptual once the mathematical foundations are established. For now, you can think of the following example: 

* One wishes to fit some data points to a best fit linear line $$y=ax+b$$. One chooses initial "guesses" for $$a$$ and $$b$$ and a machine learning algorithm optimizes for the optimal values of $$a$$ and $$b$$ with respect to the mean-squared error.

There are three main tasks in machine learning:

1. **Regression**. The input is multi-dimensional data points and the output is a **real number** (or sequence of real numbers). For example, the input might be height and weight and the corresponding output resting heart rate and resting blood pressure (a sequence of real numbers).

2. **Classification**. The input is multi-dimensional data points and the output is an **integer** (which represents different classes). Consider the following example with two classes: pictures of roses and pictures of begonias. The input would be multi-dimensional images (color channel included) and one may assign the integer 0 to roses and the integer 1 to begonias. 

3. **Generation**: The input is noise and the output is something sensible. A few concrete examples include training a machine learning algorithm to take in a *random seed* and generate images of peoples faces.

The goal of machine learning is to create a function which is fed the input and spits out the output. There **are many many** different functions one can use and in general they will look very different depending on the data set used.


# What Role Does Machine Learning have in Particle Physics?

* One wants to classify detected particles as signal or background events based on their energy, momentum, charge, etc... . This specific problem was featured on Kaggle 6 years ago [here](https://www.kaggle.com/c/higgs-boson/data). This problem will also be examined in this tutorial.
* (My Research) Particle energy is measured in the liquid argon calotimeter by studying the output current of the electronics. When a particle deposits energy in a cell, a unique pulse shape is formed. One wants to determine the energy of the incident particles based on the form of the current time series. The traditional approach was to use a signal processing technique known as the optimal filter, but as the lumonisity of the LHC increases, these pulse shapes are littered with more and more deposited energy from background events.

Machine learning has become quite popular in scientific fields in the past few years. The following plot from [MIT Technical Review](https://www.technologyreview.com/2019/01/25/1436/we-analyzed-16625-papers-to-figure-out-where-ai-is-headed-next/) examines ll of the papers available in the “artificial intelligence” section of arXiv through November 18, 2018.

![ML Popularity](../plots/ml_populatir.PNG){:width="80%"}

The following are a few recent articles on machine learning in particle physics

* [Machine and Deep Learning Applications in Particle Physics](https://arxiv.org/abs/1912.08245)
* [Machine learning at the energy and intensity frontiers of particle physics](https://www.nature.com/articles/s41586-018-0361-2)


# Where Should I Go if I Want to Get Good At Machine Learning

Machine Learning is not something you'll learn in an hour. It's a skill you need to develop over time, and like any skill you need to practice a little bit every day. If you want to reall *excel* at machine learning, my recommendation is to order [this book](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) and to **read and code along with each chapter**. Don't go crazy: just do 30 minutes a day. You'd be surprised how much you could learn in a couple months. 

My list of machine learning resources, ranked:

1. [Hands on Machine Learning; Geron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
2. Alternatively, you could also try an online course such as [this one](https://www.coursera.org/learn/machine-learning)


{% include links.md %}

