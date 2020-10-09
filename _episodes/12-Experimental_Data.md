---
title: "Applying To Experimental Data"
teaching: 5
exercises: 15
questions:
- "What about real, experimental data?"
- "Are we there yet?"
objectives:
- "Check that our machine learning models behave similarly with real experimental data."
- "Finish!"
keypoints:
- "It's a good idea to check whether our machine learning models behave well with real experimental data."
- "That's it!"
---

<iframe width="427" height="251" src="https://www.youtube.com/embed?v=GbedkKJiGq4&list=PLKZ9c4ONm-VmHsMKImIDEMsZI1Vp0UY-Z&index=10&ab_channel=HEPSoftwareFoundation" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# What about *real, experimental* data?

Notice that we've trained and tested our machine learning models on simulated data for signal and background. That's why there are definite labels, `y`. This has been a case of **supervised learning** since we knew the labels (y) going into the game. Your machine learning models would then usually be *applied* to real experimental data once you're happy with them.

To make sure our machine learning model makes sense when applied to real experimental data, we should check whether simulated data and real experimental data have the same shape in classifier threshold values.

We first need to get the real experimental data.

> ## Challenge to end all challenges
> 1. Read data.csv like in the [Data Discussion lesson](https://hsf-training.github.io/hsf-training-ml-webpage/06-Data_Discussion/index.html)
> 2. Apply cut_lep_type and cut_lep_charge like in the [Data Discussion lesson](https://hsf-training.github.io/hsf-training-ml-webpage/06-Data_Discussion/index.html)
> 3. Convert the data to a NumPy array, `X_data`, similar to the [Data Preprocessing lesson](https://hsf-training.github.io/hsf-training-ml-webpage/07-Data_Preprocessing/index.html). You may find the attribute `.values` useful to convert a pandas DataFrame to a Numpy array.
> 4. Don't forget to transform using the scaler like in the [Data Preprocessing lesson](https://hsf-training.github.io/hsf-training-ml-webpage/07-Data_Preprocessing/index.html)
> 5. Predict the labels your random forest classifier would assign to `X_data`. Call your predictions `y_data_RF`.
>
> > ## Solution
> > ~~~
> > DataFrames['data'] = pd.read_csv('/kaggle/input/4lepton/data.csv') # read data.csv file
> > DataFrames['data'] = DataFrames['data'][ np.vectorize(cut_lep_type)(DataFrames['data'].lep_type_0,
> >                                                                     DataFrames['data'].lep_type_1,
> >                                                                     DataFrames['data'].lep_type_2,
> >                                                                     DataFrames['data'].lep_type_3) ]
> > DataFrames['data'] = DataFrames['data'][ np.vectorize(cut_lep_charge)(DataFrames['data'].lep_charge_0,
> >                                                                       DataFrames['data'].lep_charge_1,
> >                                                                       DataFrames['data'].lep_charge_2,
> >                                                                       DataFrames['data'].lep_charge_3) ]
> > X_data = DataFrames['data'][ML_inputs].values # .values converts straight to NumPy array
> > X_data = scaler.transform(X_data) # X_data now scaled same as training and testing sets
> > y_data_RF = RF_clf.predict(X_data) # make predictions on the data
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}

Now we can overlay the real experimental data on the simulated data.

~~~
thresholds = [] # define list to hold random forest classifier probability predictions for each sample
for s in samples: # loop over samples
    thresholds.append(RF_clf.predict_proba(scaler.transform(DataFrames[s][ML_inputs]))[:,1]) # get ML_inputs from DataFrames[s], transform the values, predict probabilities
plt.hist(thresholds, bins=np.arange(0, 0.8, 0.1), density=True, stacked=True) # plot simulated data
data_hist = np.histogram(RF_clf.predict_proba(X_data)[:,1], bins=np.arange(0, 0.8, 0.1), density=True)[0] # histogram the experimental data
scale = sum(RF_clf.predict_proba(X_data)[:,1]) / sum(data_hist) # get scale imposed by density=True
data_err = np.sqrt(data_hist * scale) / scale # get error on experimental data
plt.errorbar(x=np.arange(0.05, 0.75, 0.1), y=data_hist, yerr=data_err) # plot the experimental data errorbars
plt.xlabel('Threshold')
~~~
{: .language-python}

Within errors, the real experimental data errorbars agree with the simulated data histograms. Good news, our random forest classifier model makes sense with real experimental data!

> ## Ok maybe one more challenge...
> In a new cell, make the same plot for your neural network classifier. Change the `np.arange` calls to (0, 0.6, 0.1), (0, 0.6, 0.1) and (0.05, 0.55, 0.1) respectively. Do real experimental data agree with simulated data in this case?
>
> > ## Solution
> > ~~~
> > thresholds = [] # define list to hold random forest classifier probability predictions for each sample
> > for s in samples: # loop over samples
> >     thresholds.append(NN_clf.predict_proba(scaler.transform(DataFrames[s][ML_inputs]))[:,1]) # get ML_inputs from DataFrames[s], transform the values, predict probabilities
> > plt.hist(thresholds, bins=np.arange(0, 0.6, 0.1), density=True, stacked=True) # plot simulated data
> > data_hist = np.histogram(NN_clf.predict_proba(X_data)[:,1], bins=np.arange(0, 0.6, 0.1), density=True)[0] # histogram the experimental data
> > scale = sum(NN_clf.predict_proba(X_data)[:,1]) / sum(data_hist) # get scale imposed by density=True
> > data_err = np.sqrt(data_hist * scale) / scale # get error on experimental data
> > plt.errorbar(x=np.arange(0.05, 0.55, 0.1), y=data_hist, yerr=data_err) # plot the experimental data errorbars
> > plt.xlabel('Threshold')
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}


# At the end of the day

How many signal events is the random forest classifier predicting?

~~~
print(np.count_nonzero(y_data_RF==1)) # signal
~~~
{: .language-python}

What about background?

~~~
print(np.count_nonzero(y_data_RF==0)) # background
~~~
{: .language-python}

The random forest classifier is *predicting* how many real data events are signal and how many are background, how cool is that?!

> # Ready to machine learn to take over the world!
> Hopefully you've enjoyed this brief discussion on machine learning! Try playing around with the hyperparameters of your random forest and neural network classifiers, such as the number of hidden layers and neurons, and see how they effect the results of your classifiers in Python!
{: .callout}
