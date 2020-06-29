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

For this tutorial we will be using (binder)[https://mybinder.org/]. Binder allows one to run any jupyter notebook from a github repositories (as long as the repository is set up with a requirments.txt file). The following link is the link we will be using:

* https://mybinder.org/v2/gh/lukepolson/HEP_ML_Lessons_Code/624701a01e261ac0183217341239c7abcd29a948


# Models

In this section we will examine 3 different machine learning models $$f$$ for classification: the support vector classifier (SVC) the random forest (RF) and the fully connected neural network (NN).


## Random Forest
A random forest (Chapter 7) is based off of the predictions of decision trees (Chapter 6). For this particular model, there is not a concept of a loss function; training is completed using the *Classification and Regression Tree* (CART) algorithm to train decision trees, and then decision trees are used together to make predictions: this is known as a random forest. They can be trained in sci-kit learn as follows

~~~
from sklearn.ensemble import RandomForestClassifier

RF_clf = RandomForestClassifier(criterion='gini', max_depth=8, n_estimators=30)
RF_clf.fit(X_train, y_train)
y_pred = SVC_clf.predict(X_test)

# See how well the classifier does
print(accuracy_score(y_test, y_pred))
~~~
{: .language-python}

Note that the code is very similar to the SVM. In this situation we have three hyperparameters specified: `criterion`, `max_depth`, and `n_estimators`. There are other parameters the the Random Forest optimizes during training. It should have a similar accuracy score to the SVC.

## Neural Network
A neural network is a very complex model with many hyperparameters. It is unlikely you will understand the code for how the network is built today, but if you are interested in neural networks, I would highly recommend reading Chapter 10 of the text (and Chapters 11-18 as well, for that matter). A neural network can be built as follows:

~~~
import tensorflow as tf
from tensorflow import keras


def build_model(n_hidden=1, n_neurons=5, learning_rate=1e-3):
    # Build
    model = keras.models.Sequential()
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(2, activation='softmax'))
    # Compile
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
~~~
{: .language-python}

For now, ignore all the complicated hyperparameters, but note that the loss used is `sparse_categorical_crossentropy`; the neural network uses gradient descent to optimize its parameters (in this case these parameters are *neuron weights*). The network can be trained as follows:

~~~
X_valid, X_train_nn = X_train[:100], X_train[100:]
y_valid, y_train_nn = y_train[:100], y_train[100:]

NN_clf = keras.wrappers.scikit_learn.KerasClassifier(build_model)
NN_clf.fit(X_train, y_train, validation_data=(X_valid, y_valid))
y_pred = NN_clf.predict(X_test)

# See how well the classifier does
print(accuracy_score(y_test, y_pred))
~~~
{: .language-python}

The neural network should also have a similar accuracy score to the random forest and support vector machine. 
