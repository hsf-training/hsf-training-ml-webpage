---
title: "Code Example"
teaching: 5
exercises: 10
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

The data set we will be using for this example is the titanic dataset from kaggle. Data can be downloaded at https://www.kaggle.com/c/titanic/data. 

# Setting up the data set for machine learning

In the previous section I spoke about $$(x_i,y_i)$$ being the data set. Here I will show how to format the data set so we can input it to a machine learning function in sci-kit learn.


