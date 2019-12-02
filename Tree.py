"""Decision tree definition."""

import numpy as np

from utils import split_search, split


class Node:
    def __init__(self, depth=0):
        """Node definition.

        Args:
            depth(int): Depth of this node (root node depth should be 0).
        """
        self._feature_idx = None  # Feature index to use for splitting
        self._split_value = None
        self._leaf = False
        self._label = None
        self._left_child = None
        self._right_child = None
        self._depth = depth

    def train(self, x, y, feature_search=None,
              max_depth=8, min_samples_split=2, min_samples_leaf=1):
        """Training procedure for node.

        Args:
            x(ndarray): Inputs.
            y(ndarray): Labels.
            feature_search(int): Number of features to search for splitting.
            max_depth(int): Max depth of tree.
            min_samples_split(int): Number of samples in a node to allow
            split search.
            min_samples_leaf(int): Number of samples to be deemed a leaf node.
        """
        if self._depth < max_depth and x.shape[0] > min_samples_split:

            # Retrieve best split coordinates based on gini impurity
            # and two groups
            self._feature_idx, self._split_value, group_1, group_2 = \
                split_search(x, y, min_samples_leaf, feature_search)

            if self._feature_idx is not np.NaN:
                # Recursively split and train child nodes
                self._left_child = Node(self._depth + 1)
                self._right_child = Node(self._depth + 1)
                self._left_child.train(*group_1, feature_search, max_depth,
                                       min_samples_split,
                                       min_samples_leaf)
                self._right_child.train(*group_2, feature_search, max_depth,
                                        min_samples_split,
                                        min_samples_leaf)
            else:
                # Impossible to split. Convert to leaf node
                # This will occur when observations are
                # identical in a given node
                self._sprout(y)
        else:
            # End condition met. Convert to leaf node
            self._sprout(y)

    def _sprout(self, y):
        """Flag node as a leaf node."""
        self._leaf = True

        # Count classes in current node to determine class
        _classes, counts = np.unique(y, return_counts=True)
        self._label = _classes[np.argmax(counts)]

    def eval(self, x, y):
        """Return number of correct predictions over a dataset."""
        if self._leaf:
            return np.sum(y == self._label)
        else:
            group_1, group_2 = split(x, y,
                                     self._feature_idx, self._split_value)
            return self._left_child.eval(*group_1) \
                   + self._right_child.eval(*group_2)

    def count(self):
        """Recursively count nodes."""
        if self._leaf:
            return 1
        return 1 + self._left_child.count() + self._right_child.count()

    def predict(self, x):
        """Recursively predict class for a single individual.

        Args:
            x(ndarray): A single individual.

        Returns:
            (int): Class index.
        """
        if self._leaf:
            return self._label
        else:
            if x[self._feature_idx] < self._split_value:
                return self._left_child.predict(x)
            else:
                return self._right_child.predict(x)


class Tree:
    def __init__(self, max_depth=5,
                 min_samples_split=2, min_samples_leaf=1, bootstrap=False):
        """Decision tree for classification.

        Args:
            max_depth(int): Max depth of tree.
            min_samples_split(int): Number of samples in a node to allow
            split search.
            min_samples_leaf(int): Number of samples to be deemed a leaf node.
            bootstrap(boolean): Resample dataset with replacement
        """
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._bootstrap = bootstrap

        # Root node
        self._root = Node()

    def train(self, x, y, feature_search=None):
        """Training routine for tree.

        Args:
            x(ndarray): Inputs.
            y(ndarray): Labels.
            feature_search(int): Number of features to search
            during split search.

        Returns:
            None

        """
        if self._bootstrap:
            # Resample with replacement
            bootstrap_indices = np.random.randint(0, x.shape[0], x.shape[0])
            x, y = x[bootstrap_indices], y[bootstrap_indices]

        self._root.train(x, y, feature_search,
                         self._max_depth, self._min_samples_split,
                         self._min_samples_leaf)

    def eval(self, x, y):
        """Return error on dataset"""
        return 100 * (1 - self._root.eval(x, y) / x.shape[0])

    def node_count(self):
        """Count nodes in tree."""
        return self._root.count()

    def predict(self, x):
        """Predict class for one observation.

        Args:
            x(ndarray): A single observation.

        Returns:
            (int): Predicted class index.

        """
        return self._root.predict(x)
