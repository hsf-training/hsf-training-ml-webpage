---
title: "Data Discussion and Preprocessing"
teaching: 15
exercises: 10
questions:
- "What dataset is being used?"
- "How must we organize our data such that it can be used in the machine learning libraries?"
objectives:
- "Briefly describe the dataset."
- "Prepare the dataset for machine learning."
keypoints:
- "One must properly format data before any machine learning takes place."
- "Data can be formatted using scikit-learn functionality; using it effectively may take time to master."
---

# Dataset Used

The dataset we will use in this tutorial is simulated ATLAS data. Each event corresponds to 4 detected leptons: some events correspond to a Higgs Boson decay and others do not (background). Various physical quantities such as lepton charge and transverse momentum are recorded for each event. The analysis in this tutorial loosely follows [the discovery of the Higgs Boson](https://www.sciencedirect.com/science/article/pii/S037026931200857X).


# Setting up the dataset for machine learning

Here we will format the dataset $$(x_i, y_i)$$ so we can use it for machine learning with scikit-learn and TensorFlow. First, we need to open our data set and separate it into a training and test set.

~~~
df = pd.read_pickle('../input/atlas-bootcamp-2020-machine-learning-data/data.pkl')
df_train = df.iloc[0:800000]
df_test = df.iloc[800000::]
df.head()
~~~
{: .language-python}

 The data type is currently a pandas DataFrame: we now need to convert it into a NumPy array so that it can be used in scikit-learn and TensorFlow during the machine learning process. Note that there are many ways that this can be done: in this tutorial we will use the scikit-learn **pipeline** functionality to format our data set. For more information, please see [Chapter 2 of Geron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)(pg 70-71) and [the scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). We will briefly walk through the code in this tutorial.

First, we import all required modules from scikit-learn.
~~~
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
~~~
{: .language-python}

Next, a special `DataFrameSelector` class is defined. This class is defined such that it can operate with scikit-learn **pipeline** functionality; specifically, *it takes in a pandas DataFrame and outputs a NumPy array*. Some essential class methods are defined here such that the class can operate in the OOP framework of scikit-learn: specifically, the `fit` and `transform` methods are defined. For detailed information on **transformers** in scikit-learn see [Chapter 2 of Geron](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/) or [the scikit-learn documentation](https://scikit-learn.org/stable/data_transforms.html).

~~~
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
~~~
{: .language-python}

Now a **pipeline** is created. This takes in `df_train` or `df_test` and spits out a NumPy array consisting of only the DataFrame columns corresponding to headers in `attribs`. Note also the `StandardScaler`: this ensures that all numerical attributes are scaled to have a mean of 0 and a standard deviation of 1 before they are fed to the machine learning model. This type of preprocessing is common before feeding data into machine learning models and is especially important for neural networks.

~~~
attribs = ['lep_pt_1', 'lep_pt_2', 'mllll']
pipeline = Pipeline([
    ('selector', DataFrameSelector(attribs)),
    ('std_scaler', StandardScaler())
])

~~~
{: .language-python}

Now we will use the pipeline to generate the subset $$(x_i, y_i)$$ used for training and the subset $$(x_i, y_i)$$ used for testing the model. Note that `fit_transform` is called on the training dataset but `transform` is called on the test dataset. We keep the labels $$x_i$$ separate from the targets (i.e. signal/background) $$y_i$$.

~~~
X_train = pipeline.fit_transform(df_train)
y_train = df_train['type'].values

X_test = pipeline.transform(df_test)
y_test = df_test['type'].values
~~~
{: .language-python}


Now we are ready to examine various models $$f$$ for predicting whether an event corresponds to a Higgs decay or a background event.
