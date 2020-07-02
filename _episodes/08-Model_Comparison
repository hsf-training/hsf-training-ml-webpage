---
title: "Code Example"
teaching: 5
exercises: 10
questions:
- "How do you use the sci-kit learn and tensorflow packages for machine learning?"
objectives:
- "Train a support vector machine for classification."
- "Train a random forest for classification."
- "Train a simple neural network for classification."
keypoints:
- "The basic features of sci-kit learn and tensorflow are very simple to use."
- "To perform more sophisticated model construction, one should carefully read the textbook."
---

# Alternative Metrics

As seen in the previous section, accuracy is typically not the preferred metric for classifiers. In this section we will define some new metrics. Let TP, FP, TN, FN be the number of true positives, false positives, true negatives, and false negatives classified using a given model.

## Precision

Precision is defined as

$$\text{precision}=\frac{\text{TP}}{\text{TP}+\text{FP}}$$




~~~
decisions_nn = NN_clf.predict_proba(X_test)[:,1]
decisions_rf = RF_clf.predict_proba(X_test)[:,1]
fpr_nn, tpr_nn, thresholds = roc_curve(y_test, decisions_nn)
fpr_rf, tpr_rf, thresholds = roc_curve(y_test, decisions_rf)
~~~
{: .language-python}

~~~
plt.plot(fpr_rf,tpr_rf, label='Random Forest')
plt.plot(fpr_nn,tpr_nn, color='r', ls='--', label='Neural Network')
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.grid()
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
~~~
{: .language-python}

