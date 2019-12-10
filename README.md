# np-random-forest
Random Forest and Decision Tree for classification. Implemented using NumPy.

## Prerequisites

- Python 3
- Numpy
- Pandas (only for analyzing results from the demo)
- [data_fetcher package](https://github.com/sachaMorin/dataset_fetcher)

## Benchmarks: Banknote and Sonar (Mines vs. Rocks) Datasets
The training set held 80 % of samples. The remaining 20 % were used as a holdout set to compute test accuracy.

Three models were tested, consisting of 1, 64 and 128 trees respectively. Maximum depth was set to 8 for all forests.
min_samples_split and min_samples_leaf were left to their default values, that is, 2 and 1. The idea was to allow
individual trees to overfit and confirm that larger forests would lead to better generalization.

Bootstrapping was allowed for all but the single-tree forest.

The feature_search argument determines the search space when splitting a node. The single-tree model was set 
to search all features whereas the other two models were restricted to randomly search the floored square root of the total number of 
features (2 for Banknote and 7 for Sonar). This is effectively attribute selection at node-level.

Booststrapping, restricted search space for splitting nodes and the large amount of learners in a random forest should
all improve performance on the holdout set.

All experiments were run 10 times and averaged. The average test accuracy is as follows :

Number of Trees | 1 | 64 | 128
----------|-------:|---------:|---:
Banknote  |98.47 % |  99.78 % | 99.64 %
Sonar     |74.63 % |  87.56 % | 87.07 %

The record test accuracy for each model over both datasets are the following:

Number of Trees | 1 | 64 | 128
----------|-------:|---------:|---:                      
Banknote  |98.91 % |  100.00 % | 99.64 %
Sonar     |80.49 % |  90.24 % | 90.24 %
## Usage
The data_fetcher repo should be cloned in the main directory.

The package can be used to download and dump to a pickle file the [Banknote dataset](http://archive.ics.uci.edu/ml/machine-learning-databases/00266/) and the 
[Connectionist Bench (Sonar, Mines vs. Rocks) dataset](http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks))
from the UCI Machine Learning Repository.

Datasets can be retrieved like so:
```python
from dataset_fetcher.loader import load_dataset

x_train, y_train, x_test, y_test, classes = load_dataset('Banknote')

# OR

x_train, y_train, x_test, y_test, classes = load_dataset('Sonar')
```

Here's a typical use of the Forest module:
```python
from Forest import Forest

forest = Forest(max_depth=8, no_trees=128,
                min_samples_split=2, min_samples_leaf=1,
                feature_search=7, bootstrap=True)

forest.train(x_train, y_train)

train_acc = forest.eval(x_train, y_train)  # Retrieve train accuracy
test_acc = forest.eval(x_test, y_test)  # Retrieve test accuracy
```






## Acknowledgements
I learned a lot about decision trees reading Jason Brownlee's tutorial, available [here](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#links).

