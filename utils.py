"""Utilities."""

import numpy as np


def gini(*groups):
    """ Gini impurity for classification problems.

    Args: groups (tuple): tuples containing:
        (ndarray): Group inputs (x).
        (ndarray): Group labels (y).

    Returns:
        (float): Gini impurity index.

    """
    m = np.sum([group[0].shape[0] for group in groups])  # Number of samples

    gini = 0.0

    for group in groups:
        y = group[1]
        group_size = y.shape[0]

        # Count number of observations per class
        _, class_count = np.unique(y, return_counts=True)
        proportions = class_count / group_size
        weight = group_size / m

        gini += (1 - np.sum(proportions ** 2)) * weight

    return gini


def split(x, y, feature_idx, split_value):
    """ Returns two tuples holding two groups resulting from split.

    Args:
        x (ndarray): Input.
        y (ndarray): Labels.
        feature_idx (int): Feature to consider.
        split_value (float): Value used to split.

    Returns:
        (tuple):tuple containing:
            (tuple):tuple containing:
                (ndarray): Inputs of group under split.
                (ndarray): Labels of group under split.
            (tuple):tuple containing:
                (ndarray): Inputs of group over split.
                (ndarray): Labels of group over split.

    """
    bool_mask = x[:, feature_idx] < split_value
    group_1 = (x[bool_mask], y[bool_mask])
    group_2 = (x[bool_mask == 0], y[bool_mask == 0])
    return group_1, group_2


def legal_split(*groups, min_samples_leaf=1):
    """Test if all groups hold enough samples to meet the min_samples_leaf
    requirement """
    for g in groups:
        if g[0].shape[0] < min_samples_leaf:
            return False
    return True


def split_search_feature(x, y, feature_idx, min_samples_leaf):
    """Return best split on dataset given a feature.

    Return error values (np.Nan for floats and None for tuples) if no
    split can be found.

    Args:
        x(ndarray): Inputs.
        y(ndarray): Labels.
        feature_idx(int): Index of feature to consider
        min_samples_leaf(int): Minimum number of samples to be deemed
        a leaf node.

    Returns:
        (tuple):tuple containing:
            (float): gini score.
            (float): value used for splitting.
            (tuple):tuple containing:
                (tuple):tuple containing:
                    (ndarray): Inputs of group under split.
                    (ndarray): Labels of group under split.
                (tuple):tuple containing:
                    (ndarray): Inputs of group over split.
                    (ndarray): Labels of group over split.

    """
    gini_scores = []
    splits = []
    split_values = []
    series = x[:, feature_idx]

    # Greedy search on all input values for relevant feature
    for split_value in series:
        s = split(x, y, feature_idx, split_value)

        # Test if groups hold enough samples, otherwise keep searching
        if legal_split(*s, min_samples_leaf=min_samples_leaf):
            gini_scores.append(gini(*s))
            splits.append(s)
            split_values.append(split_value)

    if len(gini_scores) is 0:
        # Impossible to split
        # This will occur when samples are identical in a given node
        return np.NaN, np.NaN, None

    arg_min = np.argmin(gini_scores)

    return gini_scores[arg_min], split_values[arg_min], splits[arg_min]


def split_search(x, y, min_samples_leaf, feature_search=None):
    """Return best split on dataset.

    Return error values (np.Nan for floats and None for tuples) if no
    split can be found.

    Args:
        x(ndarray): Inputs.
        y(ndarray): Labels.
        feature_search(int): Number of features to use for split search
        min_samples_leaf(int): Minimum number of samples to be deemed
        a leaf node.

    Returns:
        (tuple):tuple containing:
            (int): Index of best feature.
            (float): value used for splitting.
            (tuple):tuple containing:
                (ndarray): Inputs of group under split.
                (ndarray): Labels of group under split.
            (tuple):tuple containing:
                (ndarray): Inputs of group over split.
                (ndarray): Labels of group over split.

    """
    gini_scores = []
    splits = []
    split_values = []

    # Flag to handle cases where no legal splits can be found
    split_flag = False

    if feature_search is None:
        # Default to all features
        feature_indices = np.arange(x.shape[1])
    else:
        if feature_search > x.shape[1]:
            raise Exception('Tried searching more features than '
                            'available features in dataset.')

        # Randomly choose feature_search features to look up
        feature_indices = np.random.choice(x.shape[1],
                                           feature_search,
                                           replace=False)

    # Search over features
    for feature_idx in feature_indices:
        g, s_value, s = split_search_feature(x, y,
                                             feature_idx, min_samples_leaf)
        gini_scores.append(g)
        split_values.append(s_value)
        splits.append(s)

        if g is not np.NaN:
            # At least one legal split
            split_flag = True

    if not split_flag:
        # Impossible to split
        # This will occur when samples are identical in a given node
        return np.NaN, np.NaN, None, None

    arg_min = np.nanargmin(gini_scores)

    group_1, group_2 = splits[arg_min]

    return feature_indices[arg_min], split_values[arg_min], group_1, group_2
