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

In a **traditional approach** to solving problems, one would study a problem, write rules (i.e. laws of physics) to solve that problem, analyze errors, then modify the rules. A **machine learning approach** automates this process: *a machine learning model modifies its rules based on the errors it measures*.

<ul>
  For example, consider `$z = x + y_i$` fitting some data \\( (x_i, y_i) \\) to a quadratic curve $y=ax^2+bx+c$. In this case, one might define the error to be the difference of squares: namely $`\sum_i (y_i-(ax_i^2+bx_i+c))`$. This error function depends on 3 input parameters $a$, $`b`$ and $`c`$. Once can tweak these parameters to **minimize the loss function**.
</ul>
{% include links.md %}

