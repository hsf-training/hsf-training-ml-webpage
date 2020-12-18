---
title: "OPTIONAL: TensorFlow"
teaching: 5
exercises: 15
questions:
- "What other machine learning libraries can I use?"
- "How do classifiers built with different libraries compare?"
objectives:
- "Train a neural network model in TensorFlow."
- "Compare your TensorFlow neural network with your other classifiers."
keypoints:
- "TensorFlow is another good option for machine learning in Python."
---


# TensorFlow

In this section we will examine the fully connected neural network (NN) in TensorFlow, rather than PyTorch.

**TensorFlow** is an end-to-end open-source platform for machine learning. It is used for building and deploying machine learning models. See [the documentation](https://www.tensorflow.org/). 

**TensorFlow** does have GPU support, therefore can be used train complicated neural network model that require a lot of GPU power. 

TensorFlow is also interoperable with other packages discussed so far, datatypes from NumPy and pandas are often used in TensorFlow.

Here we will import all the required TensorFlow libraries for the rest of the tutorial.

~~~
from tensorflow import keras # tensorflow wrapper
from tensorflow.random import set_seed # import set_seed function for TensorFlow
set_seed(seed_value) # set TensorFlow random seed
~~~
{: .language-python}

From the previous pages, we already have our data in NumPy arrays, like `X`, that can be used in TensorFlow.

## Model training

To use a neural network with TensorFlow, we modularize its construction using a function. We will later pass this function into a Keras wrapper.

~~~
def build_model(n_hidden=1, n_neurons=5, learning_rate=1e-3): # function to build a neural network model
    # Build
    model = keras.models.Sequential() # initialise the model
    for layer in range(n_hidden): # loop over hidden layers
        model.add(keras.layers.Dense(n_neurons, activation="relu")) # add layer to your model
    model.add(keras.layers.Dense(2, activation='softmax')) # add output layer
    # Compile
    optimizer = keras.optimizers.SGD(learning_rate=learning_rate) # define the optimizer
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) # compile your model
    return model
~~~
{: .language-python}

For now, ignore all the complicated hyperparameters, but note that the loss used is `sparse_categorical_crossentropy`; the neural network uses gradient descent to optimize its parameters (in this case these parameters are *neuron weights*).

We need to keep some events for validation. Validation sets are used to select and tune the final neural network model.

~~~
X_valid_scaled, X_train_nn_scaled = X_train_scaled[:100], X_train_scaled[100:] # first 100 events for validation
~~~
{: .language-python}

> ## Challenge
> Assign the first 100 events in y_train for validation (`y_valid`) and the rest for training your neural network (`y_train_nn`).
> 
> > ## Solution
> > ~~~
> > y_valid, y_train_nn = y_train[:100], y_train[100:] # first 100 events for validation
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}

With these parameters, the network can be trained as follows:

~~~
tf_clf = keras.wrappers.scikit_learn.KerasClassifier(build_model) # call the build_model function defined earlier
tf_clf.fit(X_train_nn_scaled, y_train_nn, validation_data=(X_valid_scaled, y_valid)) # fit your neural network
~~~
{: .language-python}

> ## Challenge
> Get the predicted y values for the TensorFlow neural network, `y_pred_tf`.
> Once you have `y_pred_tf`, see how well your TensorFlow neural network classifier does using accurarcy_score.
>
> > ## Solution
> >
> > ~~~
> > y_pred_tf = tf_clf.predict(X_test_scaled) # make predictions on the test data
> >
> > # See how well the classifier does
> > print(accuracy_score(y_test, y_pred_tf))
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}

The TensorFlow neural network should also have a similar accuracy score to the PyTorch neural network.

## Overfitting check

> ## Challenge
> As shown in [the lesson on Overfitting check](https://hsf-training.github.io/hsf-training-ml-webpage/10-Overfitting_Check/index.html), make the same overfitting check for your TensorFlow neural network and decide whether any overfitting is present.
>
> > ## Solution
> > ~~~
> > compare_train_test(tf_clf, X_train_nn_scaled, y_train_nn, X_test_scaled, y_test, 'TensorFlow Neural Network output')
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}

## Model Comparison

> ## Challenge
> Print the same classification report as [the Model Comparison lesson](https://hsf-training.github.io/hsf-training-ml-webpage/11-Model_Comparison/index.html) for your TensorFlow neural network.
>
> > ## Solution
> > ~~~
> > # TensorFlow Neural Network Report
> > print (classification_report(y_test, y_pred_tf,
> >                             target_names=["background", "signal"]))
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}

> ## Challenge
> As in [the Model Comparison lesson](https://hsf-training.github.io/hsf-training-ml-webpage/11-Model_Comparison/index.html), get the decisions of the TensorFlow neural network classifier, `decisions_tf`.
>
> > ## Solution
> > ~~~
> > decisions_tf = tf_clf.predict_proba(X_test_scaled)[:,1] # get the decisions of the TensorFlow neural network
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}


### ROC curve

> ## Challenge
> Get the FPRs, TPRs and thresholds for the TensorFlow neural network classifier (`fpr_tf`, `tpr_tf`, `thresholds_tf`), as was done in [the Model Comparison lesson](https://hsf-training.github.io/hsf-training-ml-webpage/11-Model_Comparison/index.html).
>
> > ## Solution
> > ~~~
> > fpr_tf, tpr_tf, thresholds_tf = roc_curve(y_test, decisions_tf) # get FPRs, TPRs and thresholds for TensorFlow neural network
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}

> ## Challenge
> Modify your ROC curve plot from [the Model Comparison lesson](https://hsf-training.github.io/hsf-training-ml-webpage/11-Model_Comparison/index.html) to include another line for the TensorFlow neural network
>
> > ## Solution
> > ~~~
> > plt.plot(fpr_rf, tpr_rf, label='Random Forest') # plot random forest ROC
> > plt.plot(fpr_nn, tpr_nn, linestyle='dashed', label='PyTorch Neural Network') # plot PyTorch neural network ROC
> > plt.plot(fpr_tf, tpr_tf, linestyle='dashdot', label='TensorFlow Neural Network') # plot TensorFlow neural network ROC
> > plt.plot([0, 1], [0, 1], linestyle='dotted', color='grey', label='Luck') # plot diagonal line to indicate luck
> > plt.xlabel('False Positive Rate') # x-axis label
> > plt.ylabel('True Positive Rate') # y-axis label
> > plt.grid() # add a grid to the plot
> > plt.legend() # add a legend
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}

### Threshold

> ## Challenge
> Use the function you defined in [the Model Comparison lesson](https://hsf-training.github.io/hsf-training-ml-webpage/11-Model_Comparison/index.html) to get the AMS score for your TensorFlow neural network classifier, `ams_tf`.
>
> > ## Solution
> > ~~~
> > ams_tf = AMS(tpr_tf, fpr_tf, 0.001) # get AMS for TensorFlow neural network
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}

> ## Challenge
> Modify your threshold plot from [the Model Comparison lesson](https://hsf-training.github.io/hsf-training-ml-webpage/11-Model_Comparison/index.html) to include another line for the TensorFlow neural network
>
> > ## Solution
> > ~~~
> > plt.plot(thresholds_rf, ams_rf, label='Random Forest') # plot random forest AMS
> > plt.plot(thresholds_nn, ams_nn, linestyle='dashed', label='PyTorch Neural Network') # plot PyTorch neural network AMS
> > plt.plot(thresholds_tf, ams_tf, linestyle='dashdot', label='TensorFlow Neural Network') # plot TensorFlow neural network AMS
> > plt.xlabel('Threshold') # x-axis label
> > plt.ylabel('AMS') # y-axis label
> > plt.title('AMS with $b_r=0.001$') # add plot title
> > plt.legend() # add a legend
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}

> ## Challenge 

All things considered, how does your TensorFlow neural network compare to your PyTorch neural network and scikit-learn random forest?

Your feedback is very welcome! Most helpful for us is if you "[Improve this page on GitHub](https://github.com/hsf-training/hsf-training-ml-webpage/edit/gh-pages/_episodes/13-TensorFlow.md)". If you prefer anonymous feedback, please [fill this form](https://forms.gle/XBeULpKXVHF8CKC17).
