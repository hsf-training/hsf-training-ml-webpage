---
title: "Introduction"
teaching: 10
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

In a **traditional approach** to solving problems, one would study a problem, write rules (i.e. laws of physics) to solve that problem, analyze errors, then modify the rules. A **machine learning approach** automates this process: *a machine learning model modifies its rules based on the errors it measures*. While this statement is abstract now, it will become more apparent once the mathematical foundations are established. For now, you can think of the following example: 

* One wishes to fit some data points to a best fit linear line $$y=ax+b$$. One chooses initial "guesses" for $$a$$ and $$b$$ and a machine learning algorithm optimizes for the optimal values of $$a$$ and $$b$$ with respect to the mean-squared error.

There are two main tasks in machine learning:

1. **Regression**. The input is multi-dimensional data points and the output is a **real number**. For example, the input might be height and weight and the corresponding output resting heart rate (a real number).

2. **Classification**. The input is multi-dimensional data points and the output is an **integer** (which represents different classes). Consider the following example with two classes: pictures of roses and pictures of begonias. The input would be multi-dimensional images (color channel included) and one may assign the integer 0 to roses and the integer 1 to begonias. 

The goal of machine learning is to create a function which is fed the input and spits out the output. There **are many many** different functions one can use and in general they will look very different depending on the data set used.


# What Role Does Machine Learning have in Particle Physics?

Machine learning is useful whenever you have a dataset $$(x_i, y_i)$$ and the relationship $$y=f(x)$$ is difficult to determine through a traditional approach. In such a case, one would use one of the common machine learning model functions (such as a neural network) as a model $$f$$ (which generalizes quite nicely) and then use a form of gradient descent to tune the neural network parameters. Lets examine a few cases where this shows up in particle physics

* One wants to classify detected particles as signal or background events (the $$y_i$$) based on their energy, momentum, charge, etc... (the $$x_i$$). This specific problem was featured on Kaggle 6 years ago: https://www.kaggle.com/c/higgs-boson/data. 
* (My Research) Particle energy is measured in the liquid argon calotimeter by studying the current flowing through the electronics. When a particle deposits energy in one of the cells, a unique pulse shape is seen in the current. One wants to determine the energy of the incident particles (the $$y_i$$) based on the form of the current time series (the $$x_i$$). The traditional approach was to use a signal processing technique known as the optimal filter, but as the lumonisity of the LHC increases, these pulse shapes are littered with more and more deposited energy from background events.

# Where Should I Go if I Want to Get Good At Machine Learning

Machine Learning is not something you'll learn in an hour. It's a skill you need to develop over time, and like any skill you need to practice a little bit every day. If you want to reall *excel* at machine learning, my recommendation is to order https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ and to **read and code along with each chapter**. Don't go crazy: just do 30 minutes a day. You'd be surprised how much you could learn in a couple months. 


{% include links.md %}

