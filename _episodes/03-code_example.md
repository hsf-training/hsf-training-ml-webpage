---
title: "Code Example"
teaching: 5
exercises: 10
questions:
- "What is machine learning?"
- "What role does machine learning have in particle physics?"
- "What should I do if I want to get good at machine learning?"
objectives:
- "Discuss the general learning task in machine learning."
- "Provide examples of machine learning in particle physics."
- "Give resources to people who want to become proficient in machine learning."
keypoints:
- "In general, machine learning is about designing a function $$f$$ such that $$y=f(x)$$ fits a dataset $$(x_i,y_i)$$. The domain and range of $$f$$ aren't necessarily real numbers: in fact, they are often much more complicated."
- "Machine learning has many applications in particle physics."
- "If you want to become proficient in machine learning, you need to practice."
---

The data set we will be using for this example is the titanic dataset from kaggle. Data can be downloaded at https://www.kaggle.com/c/titanic/data.

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

I'll walk through each part of the code:

1. A special `DataFrameSelector` class is defined. Don't worry that it looks complicated; all that matters is that its in a format that communicates with sci-kit learn pipeline functionality and that *it takes in a pandas DataFrame and outputs a numpy array*.  
2. A numerical pipeline is created. This takes in `df_train` or `df_test` and spits out a numpy array consisting of only the numerical attributes of the DataFrame. Note also the `StandardScaler`: this ensures that all numerical attributes are scaled to have a mean of 0 and a standard deviation of 1 before they are fed to the machine learning model. This type of preprocessing is common before feeding data into machine learning models. This is especially import for neural networks.
3. A categorical pipeline is created. This takes in `df_train` or `df_test`, selects only the categorical attributes of the DataFrame applies the `OneHotEncoder` (to create numerical values for categorical variables), and then spits out a numpy array consisting of these values.
4. The full pipeline concatenates these two numpy arrays.
5. The pipeline is used to generate the subset $$(x_i, y_i)$$ used for training and the subset $$(x_i, y_i)$$ used for testing the model.





