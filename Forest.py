"""Random Forest definition."""

import numpy as np

from Tree import Tree


class Forest:
    def __init__(self, max_depth=5, no_trees=7,
                 min_samples_split=2, min_samples_leaf=1, feature_search=None,
                 bootstrap=True):
        """Random Forest implementation using numpy.

        Args:
            max_depth(int): Max depth of trees.
            no_trees(int): Number of trees.
            min_samples_split(int): Number of samples in a node to allow
            split search.
            min_samples_leaf(int): Number of samples to be deemed a leaf node.
            feature_search(int): Number of features to search when splitting.
            bootstrap(boolean): Resample dataset with replacement
        """
        self._trees = []
        self._max_depth = max_depth
        self._no_trees = no_trees
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._feature_search = feature_search
        self._bootstrap = bootstrap

    def train(self, x, y):
        """Training procedure.

        Args:
            x(ndarray): Inputs.
            y(ndarray): Labels.

        Returns:
            None

        """
        print('Training Forest...\n')
        for i in range(self._no_trees):
            print('\nTraining Decision Tree no {}...\n'.format(i + 1))
            tree = Tree(max_depth=self._max_depth,
                        min_samples_split=self._min_samples_split,
                        min_samples_leaf=self._min_samples_leaf,
                        bootstrap=self._bootstrap)
            tree.train(x, y, feature_search=self._feature_search)
            self._trees.append(tree)

    def eval(self, x, y):
        """"Evaluate error on dataset."""
        p = self.predict(x)
        return (1 - np.sum(p == y) / x.shape[0]) * 100

    def predict(self, x):
        """Return predicted labels for given inputs."""
        return np.array([self._aggregate(x[i]) for i in range(x.shape[0])])

    def _aggregate(self, x):
        """Predict class by pooling predictions from all trees.

        Args:
            x(ndarray): A single example.

        Returns:
            (int): Predicted class index.

        """
        temp = [t.predict(x) for t in self._trees]
        _classes, counts = np.unique(np.array(temp), return_counts=True)

        # Return class with max count
        return _classes[np.argmax(counts)]

    def node_count(self):
        """Return number of nodes in forest."""
        return np.sum([t.node_count() for t in self._trees])
