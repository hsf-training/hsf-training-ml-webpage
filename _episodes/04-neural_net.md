---
title: "Neural Networks"
teaching: 5
exercises: 10
questions:
- "What is a neural network?"
- "How can I visualize a neural network?"
objectives:
- "Examine the structure of a fully connected sequential neural network."
- "Look at the tensorflow neural network playground to visualize how a neural network works."
keypoints:
- "The basic features of sci-kit learn and tensorflow are very simple to use."
- "To perform more sophisticated model construction, one should carefully read the textbook."
---

Here we will introduce the mathematics of a neural network. You are likely familiar with the linear transform $$y=Ax+b$$ where $$A$$ is a matrix (not necessarily square) and $$y$$ and $b$$ have the same dimensions and $$x$$ may have a different dimension. For example, if $$x$$ has dimension $$n$$ and $$y$$ and $$b$$ have dimensions $$m$$ then the matrix $$A$$ has dimension $$m$$ by $$n$$.

Now suppose we have some vector $$x_i$$ listing some features (height, weight, body fat) and $$y_i$$ contains blood pressure and resting heart rate. A simple linear model to predict the label features given the input features is then  $$y=Ax+b$$ or $$f(x)=Ax+b$$. But we can go further. Suppose we also apply a *simple* but *non-linear* function $$g$$ to the output so that $$f(x) = g(Ax+b)$$. 
