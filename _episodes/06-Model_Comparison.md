---
title: "Model Comparison"
teaching: 5
exercises: 15
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

As seen in the previous section, accuracy is typically not the preferred metric for classifiers. In this section we will define some new metrics. Let TP, FP, TN, FN be the number of true positives, false positives, true negatives, and false negatives classified using a given model. **Note that in this terminology, a background event is considered negative while a signal event is considered positive**.

* TP: Signal events correctly identified as signal events
* FP: Background events incorrectly identified as signal events
* TN: Background events correctly identified as background events
* FN: Signal events incorrectly identified as background events

Before getting into these metrics, it is important to note that a machine learning binary classifier is capable of providing a probability that a given instance corresponds to a signal or background (i.e. it would output `[0.2, 0.8]` where the first index corresponds to background and the second index as signal).

> ## Probability or not?
> There is some debate as to whether the numbers in the output of a machine learning classifier (such as `[0.2, 0.8]`) actually represent probabilities. For more information read [the following sci-kit learn documentation](https://scikit-learn.org/stable/modules/calibration.html). In general, for a *well calibrated classifier*, these do in fact represent probabilities in the frequentist interpretation. It can be difficult, however, to assess whether or not a classifier is indeed *well calibrated*. As such, it may be better to interpret these as confidence levels rather than probabilities.
{: .callout}

It is then up to a human user to specify the probability **threshold** at which something is classified as a signal. For example, you may want the second index to be greater than 0.999 to classify something as a signal. As such, the TP, FP, TN and FN can be altered for a given machine learning classifier based on the threshold requirement for classifying something as a signal event.

> ## Classifiers in Law
> In criminal law, Blackstone's ratio (also known as the Blackstone ratio or Blackstone's formulation) is the idea that it is better that ten guilty persons escape than that one innocent suffer. This corresponds to the minimum threshold requirement of 91% confidence of a crime being commited for the classification of guilty. It is obviously difficult to get such precise probabilities when dealing with crimes. 
{: .callout}

Since TP, FP, TN, and FN all depend on the threshold of a classifier, each of these metrics can be considered functions of threshold.

## Precision

Precision is defined as

$$\text{precision}=\frac{\text{TP}}{\text{TP}+\text{FP}}$$

It is the ratio of all things that were **correctly** classified as positive to all things that **were** classified as positive. Precision itself is an imperfect metric: a trivial way to have perfect precision is to make one single positive prediction and ensure it is correct (1/1=100%) but this would not be useful. This is equivalent to having a very high threshold. As such, precision is typically combined with another metric: recall.

## Recall (True Positive Rate)

Recall is defined as 

$$\text{recall}=\frac{\text{TP}}{\text{TP}+\text{FN}}$$

It is the ratio of all things that were **correctly** classified as positive to all things that **should have been** classified as positive. Recall itself is also an imperfect metric: a trivial way to have perfect recall is to classify everything as positive; doing so, however, would result in a poor precision score. This is equivalent to having a very low threshold. As such, precision and recall need to be considered together.


## Metrics for our Classifier

By default, the threshold was set to 50% in computing the accuracy score when we made the predictions earlier. For consistency, the threshold is set to 50% here.

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
The ROC curve is a plot of the recall (or true positive rate) vs. the false positive rate: the ratio of negative instances incorrectly classified as positive. A classifier may classify many instances as positive (i.e. has a low tolerance for classifying something as positive), but in such an example it will probably also incorrectly classify many negative instances as positive as well. The false positive rate is plotted on the x-axis of the ROC curve and the true positive rate on the y-axis; the threshold is varied to give a parameteric curve. A random classifier results in a line. Before we look at the ROC curve, let's examine the following plot

~~~
decisions_nn = NN_clf.predict_proba(X_test)[:,1]
decisions_rf = RF_clf.predict_proba(X_test)[:,1]

plt.figure()
plt.hist(decisions_nn[y_test==1], color='b', histtype='step', bins=50, label='Higgs Events')
plt.hist(decisions_nn[y_test==0], color='g', histtype='step', bins=50, label='Background Events')
plt.xlabel('Threshhold')
plt.ylabel('Number of Events')
plt.semilogy()
plt.legend()
plt.show()
~~~
{: .language-python}

We can separate this plot into two separate histograms (Higgs vs. non Higgs) because we know beforehand which events correspond to the particular type of event. For real data where the answers aren't provided, it will be one concatenated histogram. The game here is simple: we pick a threshold (i.e. vertical line on the plot). Once we choose that threshold, everything to the right of that vertical line is classified as a signal event, and everything to the left is classified as a background event. By moving this vertical line left and right (i.e. altering the threshold) we effectively change TP, FP, TN and FN. Hence we also change the true positive rate and the false positive rate by moving this line around.

Suppose we move the threshold from 0 to 1 in steps of 0.01. In doing so, we will get an array of TPRs and FPRs. We can then plot the TPR array vs. the FPR array: this is the ROC curve. To plot the ROC curve, we need to obtain the probabilities that something is classified as a signal (rather than the signal/background prediction itself). This can be done as follows:

~~~
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



We need to decide on an appropriate threshhold.

## What Should My Threshhold Be?

As discussed above, the threshhold depends on the problem at hand. In this specific example of classifying particles as signal or background events, the primary goal is optimizing the discovery region for statistical significance. As discussed [here](https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf), this metric is the approximate median significance (AMS) defined as 

$$\text{AMS} = \sqrt{2\left((s+b+b_r)\ln\left(\frac{s}{b+b_r}\right)-s \right)} $$

where $$s$$ and $$b$$ are the true and false positive rates and $$b_r$$ is some number chosen to reduce the variance of the AMS such that the selection region is not too small. For the purpose of this tutorial we will choose $$b_r=0.001$$. 

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
