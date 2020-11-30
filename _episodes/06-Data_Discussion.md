---
title: "Data Discussion"
teaching: 10
exercises: 5
questions:
- "What dataset is being used?"
- "How do some of the variables look?"
objectives:
- "Briefly describe the dataset."
- "Take a peek at some variables."
keypoints:
- "One must know a bit about your data before any machine learning takes place."
- "It's a good idea to visualise your data before any machine learning takes place."
---

<iframe width="427" height="251" src="https://www.youtube.com/embed?v=g7QGLvy9lIY&list=PLKZ9c4ONm-VmHsMKImIDEMsZI1Vp0UY-Z&index=5&ab_channel=HEPSoftwareFoundation" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# Code Example

Here we will import all the required libraries for the rest of the tutorial. All scikit-learn and PyTorch functions will be imported later on when they are required.

~~~
import pandas as pd # to store data as dataframe
import numpy as np # for numerical calculations such as histogramming
import matplotlib.pyplot as plt # for plotting
~~~
{: .language-python}

You can check the version of these packages by checking the `__version__` attribute.

~~~
np.__version__
~~~
{: .language-python}

Let's set the random seed that we'll be using. This reduces the randomness when you re-run the notebook

~~~
seed_value = 420 # 42 is the answer to life, the universe and everything
from numpy.random import seed # import the function to set the random seed in NumPy
seed(seed_value) # set the seed value for random numbers in NumPy
~~~
{: .language-python}


# Dataset Used

The dataset we will use in this tutorial is simulated ATLAS data. Each event corresponds to 4 detected leptons: some events correspond to a Higgs Boson decay (<span style="color:orange">signal</span>) and others do not (<span style="color:blue">background</span>). Various physical quantities such as lepton charge and transverse momentum are recorded for each event. The analysis in this tutorial loosely follows [the discovery of the Higgs Boson](https://www.sciencedirect.com/science/article/pii/S037026931200857X).

~~~
# In this notebook we only process the main signal ggH125_ZZ4lep and the main background llll, 
# for illustration purposes.
# You can add other backgrounds after if you wish.
samples = ['ggH125_ZZ4lep','llll']
~~~
{: .language-python}

# Exploring the Dataset

Here we will format the dataset $$(x_i, y_i)$$ so we can explore! First, we need to open our data set and read it into pandas DataFrames.

~~~
# get data from files

DataFrames = {} # define empty dictionary to hold dataframes
for s in samples: # loop over samples
    DataFrames[s] = pd.read_csv('/kaggle/input/4lepton/'+s+".csv") # read .csv file

DataFrames # print data
~~~
{: .language-python}

Before diving into machine learning, think about whether there are any things you should do to clean up your data. In the case of this Higgs analysis, Higgs boson decays should produce 4 electrons or 4 muons or 2 electrons and 2 muons. Let's define a function to keep only events which produce 4 electrons or 4 muons or 2 electrons and 2 muons. 

~~~
# cut on lepton type
def cut_lep_type(lep_type_0,lep_type_1,lep_type_2,lep_type_3):
# first lepton is [0], 2nd lepton is [1] etc
# for an electron lep_type is 11
# for a muon lep_type is 13
# only want to keep events where one of eeee, mumumumu, eemumu
    sum_lep_type = lep_type_0 + lep_type_1 + lep_type_2 + lep_type_3
    if sum_lep_type==44 or sum_lep_type==48 or sum_lep_type==52: return True
    else: return False
~~~
{: .language-python}

We then need to apply this function on our DataFrames.

~~~
# apply cut on lepton type
for s in samples:
    # cut on lepton type using the function cut_lep_type defined above
    DataFrames[s] = DataFrames[s][ np.vectorize(cut_lep_type)(DataFrames[s].lep_type_0,
                              		                      DataFrames[s].lep_type_1,
                                          	              DataFrames[s].lep_type_2,
                                                  	      DataFrames[s].lep_type_3) ]
DataFrames # print data
~~~
{: .language-python}

> ## Challenge
> Another cut to clean our data is to keep only events where lep_charge_0+lep_charge_1+lep_charge_2+lep_charge_3==0. 
> Write out this function yourself then apply it.
>
> > ## Solution
> >
> > ~~~
> > # cut on lepton charge
> > def cut_lep_charge(lep_charge_0,lep_charge_1,lep_charge_2,lep_charge_3):
> > # only want to keep events where sum of lepton charges is 0
> >     sum_lep_charge = lep_charge_0 + lep_charge_1 + lep_charge_2 + lep_charge_3
> >     if sum_lep_charge==0: return True
> >     else: return False
> >
> > # apply cut on lepton charge
> > for s in samples:
> >     # cut on lepton charge using the function cut_lep_charge defined above
> >     DataFrames[s] = DataFrames[s][ np.vectorize(cut_lep_charge)(DataFrames[s].lep_charge_0,
> >                                                     	    DataFrames[s].lep_charge_1,
> >                                                     	    DataFrames[s].lep_charge_2,
> >                                                     	    DataFrames[s].lep_charge_3) ]
> > DataFrames # print data
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}

# Plot some input variables

In any analysis searching for <span style="color:orange">signal</span> one wants to optimise the use of various input variables. Often, this optimisation will be to find the best <span style="color:orange">signal</span> to <span style="color:blue">background</span> ratio. Here we define histograms for the variables that we'll look to optimise.

~~~
lep_pt_2 = { # dictionary containing plotting parameters for the lep_pt_2 histogram
    # change plotting parameters
    'bin_width':1, # width of each histogram bin
    'num_bins':13, # number of histogram bins
    'xrange_min':7, # minimum on x-axis
    'xlabel':r'$lep\_pt$[2] [GeV]', # x-axis label
}
~~~
{: .language-python}

> ## Challenge
> Write a dictionary of plotting parameters for lep_pt_1, using 28 bins. We're using 28 bins here since lep_pt_1 extends to higher values than lep_pt_2.
>
> > ## Solution
> >
> > ~~~
> > lep_pt_1 = { # dictionary containing plotting parameters for the lep_pt_1 histogram
> >     # change plotting parameters
> >     'bin_width':1, # width of each histogram bin
> >     'num_bins':28, # number of histogram bins
> >     'xrange_min':7, # minimum on x-axis
> >     'xlabel':r'$lep\_pt$[1] [GeV]', # x-axis label
> > }
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}

Now we define a dictionary for the histograms we want to plot.

~~~
SoverB_hist_dict = {'lep_pt_2':lep_pt_2,'lep_pt_1':lep_pt_1} # add a histogram here if you want it plotted 
~~~
{: .language-python}

Now let's take a look at those variables. Because the code is a bit long, we pre-defined a function for you, to illustrate the optimum cut value on individual variables, based on <span style="color:orange">signal</span> to <span style="color:blue">background</span> ratio. Let's call the function to illustrate the optimum cut value on individual variables, based on <span style="color:orange">signal</span> to <span style="color:blue">background</span> ratio. You can check out the function definition [here](https://www.kaggle.com/meirinevans/my-functions/edit) 

We're not doing any machine learning just yet! We're looking at the variables we'll later use for machine learning.

~~~
from my_functions import plot_SoverB
plot_SoverB(DataFrames, SoverB_hist_dict)
~~~
{: .language-python}

Let's talk through the lep_pt_2 plots.
1. Imagine placing a cut at 7 GeV in the distributions of <span style="color:orange">signal</span> and <span style="color:blue">background</span> (1st plot). This means keeping all events above 7 GeV in the <span style="color:orange">signal</span> and <span style="color:blue">background</span> histograms. 
2. We then take the ratio of the number of <span style="color:orange">signal</span> events that pass this cut, to the number of <span style="color:blue">background</span> events that pass this cut. This gives us a starting value for S/B (2nd plot). 
3. We then increase this cut value to 8 GeV, 9 GeV, 10 GeV, 11 GeV, 12 GeV. Cuts at these values are throwing away more <span style="color:blue">background</span> than <span style="color:orange">signal</span>, so S/B increases. 
4. There comes a point around 13 GeV where we start throwing away too much <span style="color:orange">signal</span>, thus S/B starts to decrease. 
5. Our goal is to find the maximum in S/B, and place the cut there. You may have thought the maximum would be where the <span style="color:orange">signal</span> and <span style="color:blue">background</span> cross in the upper plot, but remember that the lower plot is the <span style="color:orange">signal</span> to <span style="color:blue">background</span> ratio of everything to the right of that x-value, not the <span style="color:orange">signal</span> to <span style="color:blue">background</span> ratio at that x-value.

The same logic applies to lep_pt_1.

In the [ATLAS Higgs discovery paper](https://www.sciencedirect.com/science/article/pii/S037026931200857X), there are a number of numerical cuts applied, not just on lep_pt_1 and lep_pt_2.

Imagine having to separately optimise about 5,6,7,10...20 variables! Not to mention that applying a cut on one variable could change the distribution of another, which would mean you'd have to re-optimise... Nightmare.

This is where a machine learning model such as a neural network can come to the rescue. A machine learning model can optimise all variables at the same time.

A machine learning model not only optimises cuts, but can find correlations in many dimensions that will give better <span style="color:orange">signal</span>/<span style="color:blue">background</span> classification than individual cuts ever could.

That's the end of the introduction to why one might want to use a machine learning model. If you'd like to try using one, just keep reading!

Your feedback is very welcome! Most helpful for us is if you "[Improve this page on GitHub](https://github.com/hsf-training/hsf-training-ml-webpage/edit/gh-pages/_episodes/06-Data_Discussion.md)". If you prefer anonymous feedback, please [fill this form](https://forms.gle/XBeULpKXVHF8CKC17).
