---
title: "Overfitting check"
teaching: 5
exercises: 5
questions:
- "How does I check whether my model has overfitted?"
objectives:
- "Determine whether your models are overfitted."
keypoints:
- "It's a good idea to check your models for overtraining."
---

# Is there any overfitting?

In this section we will check whether there has been any overfitting during the model training phase. As discussed in the [lesson on Mathematical Foundations](https://hsf-training.github.io/hsf-training-ml-webpage/02-mltechnical/index.html), overfitting can be an unwanted fly in the ointment, so it should be avoided!

Comparing a machine learning model's output distribution for the training and testing set is a popular way in High Energy Physics to check for overtraining. The compare_train_test() method will plot the shape of the Neural Network's decision function for each class, as well as overlaying it with the decision function in the training set.

There are techniques to prevent overtraining.

The code to plot the overtraining check is a bit long, so once again you can see the function definition [here](https://www.kaggle.com/meirinevans/my-functions/edit) 

~~~
from my_functions import compare_train_test
compare_train_test(RF_clf, X_train, y_train, X_test, y_test, 'Random Forest output')
~~~
{: .language-python}

The <span style="color:blue">blue</span> signal dots (test set) nicely overlap with the <span style="color:blue">blue</span> signal histogram bars (training set). The same goes for the red background. This overlap indicates that no overtaining is present. Happy days!

> ## Challenge
> Make the same overtraining check for your neural network and decide whether any overtraining is present.
> 
> > ## Solution
> > ~~~
> > compare_train_test(NN_clf, X_train_nn, y_train_nn, X_test, y_test, 'Neural Network output')
> > ~~~
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}

Now that we've checked for overtraining we can go onto comparing our machine learning models!
