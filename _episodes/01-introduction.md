---
title: "Introduction"
teaching: 10
exercises: 0
questions:
- "What is machine learning?"
- "What role does machine learning have in particle physics?"
- "Where should I start if I want to become fluent using machine learning techniques?"
objectives:
- "Discuss the possibilities and limitations of machine learning."
- "Classification"
- "Regression"
keypoints:
- "Machine learning ..."
- "can be used for ..."
- "If you want to become proficient in machine learning, sit down with the textbook ... and spend 30 mins every day coding through the book"
---

# What is Machine Learning?

General definition

<ul>
[Machine Learning is the] field of study that gives computers the ability to learn without being explicitly programmed
  <ul>
    -Arthur Samuel, 1959
  </ul>
</ul>

In a **traditional approach** to solving problems, one would study a problem, write rules (i.e. laws of physics) to solve that problem, analyze errors, then modify the rules. A **machine learning approach** automates this process: *a machine learning model modifies its rules based on the errors it measures*. We'll now define some important terms in machine learning.

  * **Data** $$(x_i, y_i)$$ where $$i$$ represents the ith data point. The $$x_i$$ are typically referred to as **instances** and the $$y_i$$ as **labels**. In general the $$x_i$$ and $$y_i$$ don't need to be numbers. For example, in a dataset consisting of pictures of animals, the $$x_i$$ might be images (consisting of height, width, and color channel) and the $$y_i$$ might be a string which states the type of animal.  
  
* **Model**: Some abstract function $$f$$ such that $$y=f(x)$$ is used to model the individual $$(x_i, y_i)$$ pairs. For example, if the $$(x_i, y_i)$$ are approximately related through a quadratic function, then an adequate model might be $$y=ax^2+bx+c$$. Note that one could also use the model $$y=ax^3+b\sin(x)+ce^x + ...$$ to model the data set; a model is just *some* function; it doesn't need to model the data well necessarily.

* **Loss Function**: A function that determines *how well the model $$y=f(x)$$ predicts the data $$(x_i, y_i)$$*. Some models work better than others. One such loss function might be $$\sum_i (y_i-f(x_i))^2$$: the mean-squared error. For the quadratic function this would be $$\sum_i (y_i-(ax_i^2+bx_i+c))^2$$. One doesn't need to use the mean-squared error as a loss function, however; one could also use the mean-absolute error, or the mean-cubed error, etc. 

![Quadratic model and data points](../plots/intro_image.png){:width="50%"}


What's important to note is that once the data and model are specified, then the loss function depends on the parameters of the model. For example, the loss function for the data set $$(x_i, y_i)$$ and the model $$y=ax^2+bx+c$$ depends on $$a$$, $$b$$, and $$c$$.

The goal of machine learning is to optimize a loss function with respect to the parameters of a model given a data set and a model. Lets suppose we have the dataset $$(x_i, y_i)$$, the model $$y=ax^2+bx+c$$, and the loss function $$L(a, b, c) = \sum_i (y_i-(ax_i^2+bx_i+c))^2$$ to determine the validity of the model. We want to minimize the loss function with respect to $$a$$, $$b$$, and $$c$$ (thus creating the best model). One such way to do this is to pick some random initial values for $$a$$, $$b$$, and $$c$$ and then do then repeat the following two steps until we reach a minimum for $$L(a,b,c)$$

1. Evaluate the gradient $$\vec{G} = (\frac{\partial L}{\partial a}, \frac{\partial L}{\partial b}, \frac{\partial L}{\partial c})$$. The negative gradients points to where the function $$L(a,b,c)$$ is decreasing.

2. Update $$(a, b, c) \to (a, b, c) - \alpha \vec{G}$$ The parameter $$\alpha$$ is known as the **learning rate** in machine learning.

This procedure is known as **gradient descent** in machine learning. It's sort of like being on a mountain, and only looking at your feet to try and reach the bottom. You'll likely move in the direction where the slope is decreasing the fastest. The problem with this technique is that it may lead you into local minima (places where the mountain has "pits" but you're note quite at the bottom). Another issue with gradient descent is that the gradient $$\vec{G} = (\frac{\partial L}{\partial a}, \frac{\partial L}{\partial b}, \frac{\partial L}{\partial c})$$ depends on *all the data points* $$(x_i, y_i)$$. This is often computationally expensive for datasets that include many data points. A solution to this is to sample a different small subset of points each time the gradient is computed. This is known as **batch gradient descent**.  

A few common functions $f$ in machine learning:
* Support Vector Machines
* Decision Trees
* Random Forests
* Neural networks (Artifical, Convolutional, Recurrent, ...)





{% include links.md %}

