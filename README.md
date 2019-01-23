# GPPrefElicit

**WARNING: This package will be updated in near future.**

GPPrefElicit is a python package (released under [MIT License](https://github.com/nawalgao/GPPrefElicit/blob/master/LICENSE)) for eliciting personalized thermal preferences of occupants 
using [GPflow](https://github.com/GPflow/GPflow), and uses [TensorFlow](http://www.tensorflow.org).

This GP-PE framework aims at finding occupant’s most preferred indoor room temperature with minimized number of survey queries to the occupants. We also add our *apriori* beliefs that the utility function is unimodal in nature. GP-PE framework elicit and learn preferences (utility functions) of occupants by: (1) maintaining a flexible representation of occupant’s utility function; (2) handle uncertainty in a principled manner; (3) select queries that allow the system to differentiate amongst the preferred states and (4) allow for incorporation of prior knowledge from different sources.

It is currently maintained by [Nimish Awalgaonkar](https://www.predictivesciencelab.org/people.html).

# Install
This package was written in `Python 2.7.14`. It is recommended to use the `Anaconda >=5.0.1` distribution, on a empty environment. The package is built on top of `gpflow 0.4.0` which has to be installed from [source]( https://github.com/GPflow/GPflow/releases/tag/0.4.0).
Then, proceed on installing the following
```
conda install numpy scipy matplotlib 
```

# Usage Example
Please check out the notebooks folder for working example and how you can use this framework by simply plugging in data you learn from new occupants.

# Contributing
If you are interested in contributing to this open source project, contact us through an issue on this repository.

# Citing GPPrefElicit

To cite GPPrefElicit, please reference **paper is currently under review**.



