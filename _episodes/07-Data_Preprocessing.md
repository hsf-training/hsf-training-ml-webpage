---
title: "Data Preprocessing"
teaching: 10
exercises: 5
questions:
- "How must we organize our data such that it can be used in the machine learning libraries?"
- "Are we ready for machine learning yet?!"
objectives:
- "Prepare the dataset for machine learning."
- "Get excited for machine learning!"
keypoints:
- "One must properly format data before any machine learning takes place."
- "Data can be formatted using scikit-learn functionality; using it effectively may take time to master."
---

<iframe width="427" height="251" src="https://www.youtube.com/embed?v=VAH0Ayha_Vc&list=PLKZ9c4ONm-VmHsMKImIDEMsZI1Vp0UY-Z&index=7&ab_channel=HEPSoftwareFoundation" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# Format the data for machine learning

It's almost time to build a machine learning model! First we choose the variables to use in our machine learning model.

~~~
ML_inputs = ['lep_pt_1','lep_pt_2'] # list of features for ML model
~~~
{: .language-python}

 The data type is currently a pandas DataFrame: we now need to convert it into a NumPy array so that it can be used in scikit-learn and TensorFlow during the machine learning process. Note that there are many ways that this can be done: in this tutorial we will use the NumPy **concatenate** functionality to format our data set. For more information, please see [the NumPy documentation on concatenate](https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html). We will briefly walk through the code in this tutorial.

~~~
#  Organise data ready for the machine learning model

# for sklearn data are usually organised
# into one 2D array of shape (n_samples x n_features)
# containing all the data and one array of categories
# of length n_samples

all_MC = [] # define empty list that will contain all features for the MC
for s in samples: # loop over the different samples
    if s!='data': # only MC should pass this
        all_MC.append(DataFrames[s][ML_inputs]) # append the MC dataframe to the list containing all MC features
X = np.concatenate(all_MC) # concatenate the list of MC dataframes into a single 2D array of features, called X

all_y = [] # define empty list that will contain labels whether an event in signal or background
for s in samples: # loop over the different samples
    if s!='data': # only MC should pass this
        if 'H125' in s: # only signal MC should pass this
            all_y.append(np.ones(DataFrames[s].shape[0])) # signal events are labelled with 1
        else: # only background MC should pass this
            all_y.append(np.zeros(DataFrames[s].shape[0])) # background events are labelled 0
y = np.concatenate(all_y) # concatenate the list of labels into a single 1D array of labels, called y
~~~
{: .language-python}

This takes in DataFrames and spits out a NumPy array consisting of only the DataFrame columns corresponding to `ML_inputs`.

Now we separate our data into a training and test set.

~~~
# This will split your data into train-test sets: 67%-33%.
# It will also shuffle entries so you will not get the first 67% of X for training
# and the last 33% for testing.
# This is particularly important in cases where you load all signal events first
# and then the background events.

# Here we split our data into two independent samples.
# The split is to create a training and testing set.
# The first will be used for classifier training and the second to evaluate its performance.

from sklearn.model_selection import train_test_split

# make train and test sets
X_train,X_test, y_train,y_test = train_test_split(X, y,
                                                  test_size=0.33,
                                                  random_state=seed_value ) # set the random seed for reproducibility
~~~
{: .language-python}

Machine learning models may have difficulty converging before the maximum number of iterations allowed if the data aren't normalized. Note that you must apply the same scaling to the test set for meaningful results (we'll apply the scaling to the test set in the next step). There are a lot of different methods for normalization of data. We will use the built-in StandardScaler for standardization. The `StandardScaler` ensures that all numerical attributes are scaled to have a mean of 0 and a standard deviation of 1 before they are fed to the machine learning model. This type of preprocessing is common before feeding data into machine learning models and is especially important for neural networks.

~~~
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # initialise StandardScaler

# Fit only to the training data
scaler.fit(X_train)
~~~
{: .language-python}

Now we will use the scaling to apply the transformations to the data.

~~~
X_train_scaled = scaler.transform(X_train)
~~~
{: .language-python}

> ## Challenge
> Apply the same scaler transformation to `X_test` and `X`.
>
> > ## Solution
> >
> > ~~~
> > X_test_scaled = scaler.transform(X_test)
> > X_scaled = scaler.transform(X)
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}

Now we are ready to examine various models $$f$$ for predicting whether an event corresponds to a <span style="color:orange">signal</span> event or a <span style="color:blue">background</span> event.

Your feedback is very welcome! Most helpful for us is if you "[Improve this page on GitHub](https://github.com/hsf-training/hsf-training-ml-webpage/edit/gh-pages/_episodes/07-Data_Preprocessing.md)". If you prefer anonymous feedback, please [fill this form](https://forms.gle/XBeULpKXVHF8CKC17).
