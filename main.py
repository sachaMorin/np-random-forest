"""Main training script."""

import numpy as np

from Forest import Forest
from dataset_fetcher.loader import load_dataset


def train_routine(dataset_name, no_trees, max_depth=8, bootstrap=None,
                  feature_search=None):
    # Load dataset
    x_train, y_train, x_test, y_test, c = load_dataset(dataset_name)

    # Number of features searched by split. Use square root as default.
    if feature_search is None:
        if no_trees > 1:
            feature_search = int(np.sqrt(x_train.shape[1]))
        else:
            feature_search = x_train.shape[1]

    if bootstrap is None:
        if no_trees > 1:
            bootstrap = True
        else:
            # Do not bootstrap if only one tree is requested
            bootstrap = False

    forest = Forest(max_depth=max_depth, no_trees=no_trees,
                    min_samples_split=2, min_samples_leaf=1,
                    feature_search=feature_search, bootstrap=bootstrap)

    forest.train(x_train, y_train)

    train_error = 1 - forest.eval(x_train, y_train)
    test_error = 1 - forest.eval(x_test, y_test)
    node_count = forest.node_count()

    print('\nMetrics :\n')
    print("train error      : {:7.4f} %"
          "\nvalidation error : {:7.4f} %"
          "\nnumber of nodes  : {:7}  "
        .format(
        train_error * 100,
        test_error * 100,
        node_count
    ))

    return train_error, test_error, node_count


# Script
if __name__ == '__main__':
    train_routine('Banknote', 1)
