---
title: "Data Discussion"
teaching: 15
exercises: 10
questions:
- "What dataset is being used?"
objectives:
- "Briefly describe the dataset."
keypoints:
- "One must properly format data before any machine learning takes place."
---

# Dataset Used

The dataset we will use in this tutorial is simulated ATLAS data. Each event corresponds to 4 detected leptons: some events correspond to a Higgs Boson decay and others do not (background). Various physical quantities such as lepton charge and transverse momentum are recorded for each event. The analysis in this tutorial loosely follows [the discovery of the Higgs Boson](https://www.sciencedirect.com/science/article/pii/S037026931200857X).

~~~
# In this notebook we only process the signal ggH125_ZZ4lep and the main background ZZ, 
# for illustration purposes.
# You can add other backgrounds after if you wish.
samples = ['ggH125_ZZ4lep','llll']
~~~
{: .language-python}

# Exploring the Dataset

Here we will format the dataset $$(x_i, y_i)$$ so we can explore! First, we need to open our data set and read it into pandas DataFrames.

~~~
# get data from files

data = {} # define empty dictionary to hold dataframes
for s in samples: # loop over samples
    data[s] = pd.read_csv('/kaggle/input/4lepton/'+s+".csv") # read .csv file
    
data # print data
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
    data[s] = data[s][ np.vectorize(cut_lep_type)(data[s].lep_type_0,
                                                  data[s].lep_type_1,
                                                  data[s].lep_type_2,
                                                  data[s].lep_type_3) ]
~~~
{: .language-python}

> ## Challenge (5 mins)
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
> >     data[s] = data[s][ np.vectorize(cut_lep_charge)(data[s].lep_charge_0,
> >                                                     data[s].lep_charge_1,
> >                                                     data[s].lep_charge_2,
> >                                                     data[s].lep_charge_3) ]
> > ~~~
> > {: .language-python}
> {: .solution}
{: .challenge}

# Plot some input variables

In any analysis searching for signal one wants to optimise the use of various input variables. Often, this optimisation will be to find the best signal to background ratio. Here we define histograms for the variables that we'll look to optimise.

~~~
lep_pt_2 = { # dictionary containing plotting parameters for the lep_pt_2 histogram
    # change plotting parameters
    'bin_width':1, # width of each histogram bin
    'num_bins':13, # number of histogram bins
    'xrange_min':7, # minimum on x-axis
    'xlabel':r'$lep\_pt$[2] [GeV]', # x-axis label
}

lep_pt_1 = { # dictionary containing plotting parameters for the lep_pt_1 histogram
    # change plotting parameters
    'bin_width':1, # width of each histogram bin
    'num_bins':28, # number of histogram bins
    'xrange_min':7, # minimum on x-axis
    'xlabel':r'$lep\_pt$[1] [GeV]', # x-axis label
}

SoverB_hist_dict = {'lep_pt_2':lep_pt_2,'lep_pt_1':lep_pt_1} 
# add a histogram here if you want it plotted
~~~
{: .language-python}

Now let's take a look at those variables. Because the code is a bit long, we pre-defined a function for you, to illustrate the optimum cut value on individual variables, based on <span style="color:blue">signal</span> to <span style="color:red">background</span> ratio. Let's call the function to illustrate the optimum cut value on individual variables, based on <span style="color:blue">signal</span> to <span style="color:red">background</span> ratio. You can check out the function definition [here](https://www.kaggle.com/meirinevans/my-functions/edit) 

We're not doing any machine learning just yet! We're looking at the variables we'll later use for machine learning.

~~~
from my_functions import plot_SoverB
plot_SoverB(data)
~~~
{: .language-python}

Let's talk through the lep_pt_2 plots.
1. Imagine placing a cut at 7 GeV in the distributions of <span style="color:blue">signal</span> and <span style="color:red">background</span> (1st plot). This means keeping all events above 7 GeV in the <span style="color:blue">signal</span> and <span style="color:red">background</span> histograms. 
2. We then take the ratio of the number of <span style="color:blue">signal</span> events that pass this cut, to the number of <span style="color:red">background</span> events that pass this cut. This gives us a starting value for S/B (2nd plot). 
3. We then increase this cut value to 8 GeV, 9 GeV, 10 GeV, 11 GeV, 12 GeV. Cuts at these values are throwing away more <span style="color:red">background</span> than <span style="color:blue">signal</span>, so S/B increases. 
4. There comes a point around 13 GeV where we start throwing away too much <span style="color:blue">signal</span>, thus S/B starts to decrease. 
5. Our goal is to find the maximum in S/B, and place the cut there.

The same logic applies to lep_pt_1.

In the [ATLAS Higgs discovery paper](https://www.sciencedirect.com/science/article/pii/S037026931200857X), there are a number of numerical cuts applied, not just on lep_pt_1 and lep_pt_2.

Imagine having to separately optimise about 5,6,7,10...20 variables! Not to mention that applying a cut on one variable could change the distribution of another, which would mean you'd have to re-optimise... Nightmare.

This is where a machine learning model such as a neural network can come to the rescue. A machine learning can optimise all variables at the same time.

A machine learning model not only optimises cuts, but can find correlations in many dimensions that will give better signal/background classification than individual cuts ever could.

That's the end of the introduction to why one might want to use a machine learning model. If you'd like to try using one, just keep reading!
