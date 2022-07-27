---
title: "Model Comparison"
teaching: 10
exercises: 10
questions:
- "How do you use the scikit-learn and PyTorch packages for machine learning?"
- "How do I see whether my machine learning model is doing alright?"
objectives:
- "Check how well our random forest is doing."
- "Check how well our simple neural network is doing."
keypoints:
- "Many metrics exist to assess classifier performance."
- "Making plots is useful to assess classifier performance."
---

<iframe width="427" height="251" src="https://www.youtube.com/embed?v=kVxz0mrTWFA&list=PLKZ9c4ONm-VmHsMKImIDEMsZI1Vp0UY-Z&index=10&ab_channel=HEPSoftwareFoundation" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# Alternative Metrics

As seen in the previous section, accuracy is typically not the preferred metric for classifiers. In this section we will define some new metrics. Let TP, FP, TN, FN be the number of true positives, false positives, true negatives, and false negatives classified using a given model. **Note that in this terminology, a <span style="color:blue">background</span> event is considered negative while a <span style="color:orange">signal</span> event is considered positive**.

* TP: <span style="color:orange">Signal</span> events correctly identified as <span style="color:orange">signal</span> events
* FP: <span style="color:blue">Background</span> events incorrectly identified as <span style="color:orange">signal</span> events
* TN: <span style="color:blue">Background</span> events correctly identified as <span style="color:blue">background</span> events
* FN: <span style="color:orange">Signal</span> events incorrectly identified as <span style="color:blue">background</span> events

Before getting into these metrics, it is important to note that a machine learning binary classifier is capable of providing a probability that a given instance corresponds to a <span style="color:orange">signal</span> or <span style="color:blue">background</span> (i.e. it would output `[0.2, 0.8]` where the first index corresponds to <span style="color:blue">background</span> and the second index as <span style="color:orange">signal</span>).

> ## Probability or not?
> There is some debate as to whether the numbers in the output of a machine learning classifier (such as `[0.2, 0.8]`) actually represent probabilities. For more information read [the following scikit-learn documentation](https://scikit-learn.org/stable/modules/calibration.html). In general, for a *well calibrated classifier*, these do in fact represent probabilities in the frequentist interpretation. It can be difficult, however, to assess whether or not a classifier is indeed *well calibrated*. As such, it may be better to interpret these as confidence levels rather than probabilities.
{: .callout}

It is then up to a human user to specify the probability **threshold** at which something is classified as a signal. For example, you may want the second index to be greater than 0.999 to classify something as a signal. As such, the TP, FP, TN and FN can be altered for a given machine learning classifier based on the threshold requirement for classifying something as a signal event.


> ## Classifiers in Law
> In criminal law, Blackstone's ratio (also known as the Blackstone ratio or Blackstone's formulation) is the idea that it is better that ten guilty persons escape than that one innocent suffer. This corresponds to the minimum threshold requirement of 91% confidence of a crime being committed for the classification of guilty. It is obviously difficult to get such precise probabilities when dealing with crimes.
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
print(classification_report(y_test, y_pred_RF, target_names=["background", "signal"]))
~~~
{: .language-python}

> ## Challenge
> Print the same classification report for your neural network.
>
> > ## Solution
> > ~~~
> > # Neural Network Report
> > print (classification_report(y_test, y_pred_NN,
> >                             target_names=["background", "signal"]))
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}

Out of the box, the random forest performs slightly better than the neural network.

Let's get the decisions of the random forest classifier.

~~~
decisions_rf = RF_clf.predict_proba(X_test_scaled)[
    :, 1
]  # get the decisions of the random forest
~~~
{: .language-python}

The decisions of the neural network classifier, `decisions_nn`, can be obtained like:

~~~
decisions_nn = (
    NN_clf(X_test_var)[1][:, 1].cpu().detach().numpy()
)  # get the decisions of the neural network
~~~
{: .language-python}

# The ROC Curve
The Receiver Operating Characteristic (ROC) curve is a plot of the recall (or true positive rate) vs. the false positive rate: the ratio of negative instances incorrectly classified as positive. A classifier may classify many instances as positive (i.e. has a low tolerance for classifying something as positive), but in such an example it will probably also incorrectly classify many negative instances as positive as well. The false positive rate is plotted on the x-axis of the ROC curve and the true positive rate on the y-axis; the threshold is varied to give a parameteric curve. A random classifier results in a line. Before we look at the ROC curve, let's examine the following plot

~~~
plt.hist(
    decisions_rf[y_test == 0], histtype="step", bins=50, label="Background Events"
)  # plot background
plt.hist(
    decisions_rf[y_test == 1],
    histtype="step",
    bins=50,
    linestyle="dashed",
    label="Signal Events",
)  # plot signal
plt.xlabel("Threshold")  # x-axis label
plt.ylabel("Number of Events")  # y-axis label
plt.semilogy()  # make the y-axis semi-log
plt.legend()  # draw the legend
~~~
{: .language-python}

We can separate this plot into two separate histograms (Higgs vs. non Higgs) because we know beforehand which events correspond to the particular type of event. For real data where the answers aren't provided, it will be one concatenated histogram. The game here is simple: we pick a threshold (i.e. vertical line on the plot). Once we choose that threshold, everything to the right of that vertical line is classified as a <span style="color:orange">signal</span> event, and everything to the left is classified as a <span style="color:blue">background</span> event. By moving this vertical line left and right (i.e. altering the threshold) we effectively change TP, FP, TN and FN. Hence we also change the true positive rate and the false positive rate by moving this line around.

Suppose we move the threshold from 0 to 1 in steps of 0.01. In doing so, we will get an array of TPRs and FPRs. We can then plot the TPR array vs. the FPR array: this is the ROC curve. To plot the ROC curve, we need to obtain the probabilities that something is classified as a <span style="color:orange">signal</span> (rather than the <span style="color:orange">signal</span>/<span style="color:blue">background</span> prediction itself). This can be done as follows:

~~~
from sklearn.metrics import roc_curve

fpr_rf, tpr_rf, thresholds_rf = roc_curve(
    y_test, decisions_rf
)  # get FPRs, TPRs and thresholds for random forest
~~~
{: .language-python}

> ## Challenge
> Get the FPRs, TPRs and thresholds for the neural network classifier (`fpr_nn`, `tpr_nn`, `thresholds_nn`).
>
> > ## Solution
> > ~~~
> > fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test, decisions_nn) # get FPRs, TPRs and thresholds for neural network
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}

Now we plot the ROC curve:

~~~
plt.plot(fpr_rf, tpr_rf, label="Random Forest")  # plot random forest ROC
plt.plot(
    fpr_nn, tpr_nn, linestyle="dashed", label="Neural Network"
)  # plot neural network ROC
plt.plot(
    [0, 1], [0, 1], linestyle="dotted", color="grey", label="Luck"
)  # plot diagonal line to indicate luck
plt.xlabel("False Positive Rate")  # x-axis label
plt.ylabel("True Positive Rate")  # y-axis label
plt.grid()  # add a grid to the plot
plt.legend()  # add a legend
~~~
{: .language-python}

*(Note: don't worry if your plot looks slightly different to the video, the classifiers train slightly different each time because they're random.)*

We need to decide on an appropriate threshold.

## What Should My Threshold Be?

As discussed above, the threshold depends on the problem at hand. In this specific example of classifying particles as <span style="color:orange">signal</span> or <span style="color:blue">background</span> events, the primary goal is optimizing the discovery region for statistical significance. As discussed [here](https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf), this metric is the approximate median significance (AMS) defined as

$$\text{AMS} = \sqrt{2\left((TPR+FPR+b_r)\ln\left(1+\frac{TPR}{FPR+b_r}\right)-TPR \right)} $$

where $$TPR$$ and $$FPR$$ are the true and false positive rates and $$b_r$$ is some number chosen to reduce the variance of the AMS such that the selection region is not too small. For the purpose of this tutorial we will choose $$b_r=0.001$$.
Other values for $$b_r$$ would also be possible. Once you've plotted AMS for the first time, you may want to play around with the value of $$b_r$$ and see how it affects your selection for the threshold value that maximizes the AMS of the plots. You may see that changing $$b_r$$ doesn't change the AMS much.

> ## Challenge
> 1. Define a function `AMS` that calculates AMS using the equation above. Call the true positive rate `tpr`, false positive rate `fpr` and $$b_r$$ `b_reg`.
> 2. Use this function to get the AMS score for your random forest classifier, `ams_rf`.
> 3. Use this function to get the AMS score for your neural network classifier, `ams_nn`.
>
> > ## Solution to part 1
> > ~~~
> > def AMS(tpr, fpr, b_reg): # define function to calculate AMS
> >     return np.sqrt(2*((tpr+fpr+b_reg)*np.log(1+tpr/(fpr+b_reg))-tpr)) # equation for AMS
> > ~~~
> > {: .language-python}
> {: .solution}
>
> > ## Solution to part 2
> > ~~~
> > ams_rf = AMS(tpr_rf, fpr_rf, 0.001) # get AMS for random forest classifier
> > ~~~
> > {: .language-python}
> {: .solution}
>
> > ## Solution to part 3
> > ~~~
> > ams_nn = AMS(tpr_nn, fpr_nn, 0.001) # get AMS for neural network
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}

Then plot:

~~~
plt.plot(thresholds_rf, ams_rf, label="Random Forest")  # plot random forest AMS
plt.plot(
    thresholds_nn, ams_nn, linestyle="dashed", label="Neural Network"
)  # plot neural network AMS
plt.xlabel("Threshold")  # x-axis label
plt.ylabel("AMS")  # y-axis label
plt.title("AMS with $b_r=0.001$")  # add plot title
plt.legend()  # add legend
~~~
{: .language-python}

*(Note: don't worry if your plot looks slightly different to the video, the classifiers train slightly different each time because they're random.)*

One should then select the value of the threshold that maximizes the AMS on these plots.

Your feedback is very welcome! Most helpful for us is if you "[Improve this page on GitHub](https://github.com/hsf-training/hsf-training-ml-webpage/edit/gh-pages/_episodes/11-Model_Comparison.md)". If you prefer anonymous feedback, please [fill this form](https://forms.gle/XBeULpKXVHF8CKC17).
