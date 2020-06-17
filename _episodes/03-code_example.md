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

The data set we will be using for this example is the titanic dataset from kaggle. The goal of this task is to predict whether or not somebody survived on the titanic based on their attributes. The data is thus as follows:

* $$x_i$$ A vector consisting of age, sex, ticket class, number of siblings/spouses aboard the titanic, number of parents/children on the titanic, ticket number, passenger fare, cabin number, and port of embarkation.
* $$y_i$$ A number representing survival (0=no and 1=yes).

We will examine 3 different machine learning models $$f$$ for classification: the support vector classifier (SVC) the random forest (RF) and the fully connected neural network (NN).  Data can be downloaded at https://www.kaggle.com/c/titanic/data.

There are two files we'll need: `train.csv` and `test.csv`. We will configure our model $$f$$ on `train.csv` and then evaluate its performance on `test.csv`. This is standard protocol in machine learning: it is possible that a model $$f$$ might work really well for modelling some training data $$(x_i,y_i)$$ (think of a very high order polynomial wiggling through every data point), but it may not generalize well to new data points. Thus we always train our model on one subset of $$(x_i, y_i)$$ and evaluate on another subset of $$(x_i, y_i)$$; this allows us to ensure that our model is not overfitting.

# Setting up the data set for machine learning

In the previous section I spoke about $$(x_i,y_i)$$ being the data set. Here I will show how to format the data set so we can input it to a machine learning function in sci-kit learn.

~~~
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('train.csv')
df.head()
~~~
{: .language-python}

There are a few problems with the way the data is stored
1. The data format is a pandas DataFrame; sci-kit learn expects numpy arrays
2. There are missing numerical values (easy to fix; insert median values)
3. There are missing categorical values (easy to fix; create new category "N" for missing values)

~~~
# Fix issue 2
df['Embarked'].fillna('N', inplace=True) 

# Define columns of dataframe corresponding to numerical and categorical attributes
num_attribs = ['Pclass','Age', 'SibSp', 'Parch', 'Fare',]
cat_attribs = ['Sex', 'Embarked']
~~~
{: .language-python}

The following procedure to turn the DataFrame into a numpy array is discussed in more detail in the textbook Hands-On Machine Learning With SciKit-Learn, Keras, and Tensorflow. The DataFrame is passed through a pipeline that turns it into a numpy array. The most noteworthy part of the pipeline is the `OneHotEncoder`. This converts anything which is a categorical attribute (such as sex: "M"/"F") into numerical values so that it they can be considered in a mathematical model

~~~
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
        
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', impute.SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', OneHotEncoder())
])

full_pipeline = FeatureUnion(transformer_list =[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])

X_train = full_pipeline.fit_transform(df_train).toarray()
y_train = df_train['Survived'].values

X_test = full_pipeline.transform(df_test).toarray()
y_test = df_test['Survived'].values
~~~
{: .language-python}

I'll walk through each part of the code:

1. A special `DataFrameSelector` class is defined. Don't worry that it looks complicated; all that matters is that its in a format that communicates with sci-kit learn pipeline functionality and that *it takes in a pandas DataFrame and outputs a numpy array*.  
2. A numerical pipeline is created. This takes in `df_train` or `df_test` and spits out a numpy array consisting of only the numerical attributes of the DataFrame. Note also the `StandardScaler`: this ensures that all numerical attributes are scaled to have a mean of 0 and a standard deviation of 1 before they are fed to the machine learning model. This type of preprocessing is common before feeding data into machine learning models. This is especially import for neural networks.
3. A categorical pipeline is created. This takes in `df_train` or `df_test`, selects only the categorical attributes of the DataFrame applies the `OneHotEncoder` (to create numerical values for categorical variables), and then spits out a numpy array consisting of these values.
4. The full pipeline concatenates these two numpy arrays.
5. The pipeline is used to generate the subset $$(x_i, y_i)$$ used for training and the subset $$(x_i, y_i)$$ used for testing the model.

Now we are ready to examine various models $$f$$ for predicting the outcome that somebody survived.

# Models

In this section we will examine 3 different machine learning models $$f$$ for classification: the support vector classifier (SVC) the random forest (RF) and the fully connected neural network (NN).

## Support Vector Classifier

The support vector machine is discussed in chapter 5 of the textbook. The following code is used fit the model $$f$$

~~~
SVC_clf = SVC(C=4.0, kernel='rbf', gamma=0.06)
SVC_clf.fit(X_train, y_train)
y_pred = SVC_clf.predict(X_test)

# See how well the classifier does
print(accuracy_score(y_test, y_pred))
~~~
{: .language-python}

* The first line creates the model $$f$$. Note that the parameters $$C=4$$, kernel, and $$\gamma=0.06$$ *are not* things that the model adjusts during training; they are known as **hyperparameters**. They are analagous to choosing the order of a polynomial when fitting a data curve; one needs to select the order $$n$$ of the polynomial they wish to fit to the data (hyperparameter), and then the coefficients $$c_0, c_1, c_2, ...$$ of $$c_0+c_1x+...+ c_nx^n$$ (parameters) are determined by gradient descent. The support vector machine has its own parameters that are adjusted using gradient descent (not discussed here). The support vector machine uses a complex loss function; the optimization problem is known as a *Quadratic Programming* (QP) problem.

* The second line fits the SVC to the training data.
* The third line makes predictions for the test data. Note that the model has not seen the test data during training. It is here that we will see if the model generalizes nicely.

You should see that the accuracy of this model is around 75-80%; it correctly predicts whether or not someone dies/survives 75-80% of the time.

## Random Forest
A random forest (Chapter 7) is based off of the predictions of decision trees (Chapter 6). For this particular model, there is not a concept of a loss function; training is completed using the *Classification and Regression Tree* (CART) algorithm to train decision trees, and then decision trees are used together to make predictions: this is known as a random forest. They can be trained in sci-kit learn as follows

~~~
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






