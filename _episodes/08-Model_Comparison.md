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

Before getting into these metrics, it is important to note that a machine learning binary classifier does not predict something as "signal" or "background" but rather gives a probability that a given instance corresponds to a signal or background (i.e. it would output `[0.271, 0.799]` where the first index corresponds to background and the second index as signal). It is then up to a human user to specify the probability **threshhold** at which something is classified as a signal. For example, you may want the second index to be greater than 0.999 to classify something as a signal. As such, the TP, FP, TN and FN can be altered for a given machine learning classifier based on the threshhold requirement for classifying something as a signal event.

> ## Classifiers in Law
> In criminal law, Blackstone's ratio (also known as the Blackstone ratio or Blackstone's formulation) is the idea that it is better that ten guilty persons escape than that one innocent suffer. This corresponds to the minimum threshhold requirement of 91% confidence of a crime being commited for the classification of guilty. It is obviously difficult to get such precise probabilities when dealing with crimes. 
{: .callout}

Since TP, FP, TN, and FN all depend on the threshhold of a classifier, each of these metrics can be considered functions of threshhold.

## Precision

Precision is defined as

$$\text{precision}=\frac{\text{TP}}{\text{TP}+\text{FP}}$$

It is the ratio of all things that were **correctly** classified as positive to all things that **were** classified as positive. Precision itself is an imperfect metric: a trivial way to have perfect precision is to make one single positive prediction and ensure it is correct (1/1=100%) but this would not be useful. This is equivalent to having a very high threshhold. As such, precision is typically combined with another metric: recall.

## Recall / True Positive Rate

Recall is defined as 

$$\text{recall}=\frac{\text{TP}}{\text{TP}+\text{FN}}$$

It is the ratio of all things that were **correctly** classified as positive to all things that **should have been** classified as positive. Recall itself is also an imperfect metric: a trivial way to have perfect recall is to classify everything as positive; doing so, however, would result in a poor precision score. This is equivalent to having a very low threshhold. As such, precision and recall need to be considered together. They can also be combined using a harmonic mean to give a metric that considers both scores.

## F1-Score

The F1-Score is defined as 

$$F_1 = \frac{2}{\frac{1}{\text{precision}}+\frac{1}{\text{recall}}} = \frac{\text{TP}}{\text{TP}+\frac{\text{FN}+\text{TP}}{2}} $$


## Metrics for our Classifier

By default, the threshhold is set to 50% when we made the predictions eariler when computing the accuracy score. As such, the threshhold is set to 50% here.

~~~
from sklearn.metrics import classification_report, roc_auc_score
# Random Forest Report
print (classification_report(y_test, y_pred_RF,
                            target_names=["background", "signal"]))
# Neural Network Report
print (classification_report(y_test, y_pred_NN,
                            target_names=["background", "signal"]))                      
~~~
{: .language-python}


# The ROC Curve
The ROC curve is a plot of the recall (or true positive rate) vs. the false positive rate: the ratio of negative instances incorrectly classified as positive. A classifier may classify many instances as positive (i.e. has a low tolerance for classifying something as positive), but in such an example it will probably also incorrectly classify many negative instances as positive as well. The ROC curve is a plot with the false positive rate on the x-axis and the true negative rate on the y-axis; the threshhold is varied to give a parameteric curve. A random classifier results in a line.

To plot the ROC curve, we need to obtain the probabilities that something is classified as a signal (rather than the signal/background prediction itself). This can be done as follows:

~~~
decisions_nn = NN_clf.predict_proba(X_test)[:,1]
decisions_rf = RF_clf.predict_proba(X_test)[:,1]
fpr_nn, tpr_nn, thresholds = roc_curve(y_test, decisions_nn)
fpr_rf, tpr_rf, thresholds = roc_curve(y_test, decisions_rf)
~~~
{: .language-python}

Now we plot the ROC curve:

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

## What Should My Threshhold Be?

As discussed above, the threshhold depends on the problem at hand. In this specific example of classifying particles as signal or background events, the primary goal is optimizing the discovery region for statistical significance. As discussed [here](https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf), this metric is the approximate median significance (AMS) defined as 

$$\text{AMS} = \sqrt{2\left((s+b+b_r)\ln\left(\frac{s}{b+b_r}\right)-s \right)} $$

where $$s$$ and $$b$$ are the true and false positive rates and $$b_r$$ is some number chosen to reduce the variance of the AMS such that the selection region is not too small. For the purpose of this tutorial we will choose $$b_r=0.001$. 

~~~
def AMS(tpr, fpr, b_reg):
    return np.sqrt(2*(tpr+fpr+b_reg)+np.log(tpr/(fpr+b_reg)) -tpr)
    
ams_nn = AMS(tpr_nn, fpr_nn, 0.001)
ams_rf = AMS(tpr_rf, fpr_rf, 0.001)
~~~
{: .language-python}

Then plot:

~~~
plt.plot(thresholds_nn, ams_nn, label='Neural Network')
plt.plot(thresholds_rf, ams_rf, label='Random Forest')
plt.xlabel('Threshhold')
plt.ylabel('AMS')
plt.title('AMS with $b_r=0.001$')
plt.legend()
plt.show()
~~~
{: .language-python}

One should then select the value of the threshhold that maximizes the AMS on these plots.
