"""Main training script."""

import numpy as np

from Forest import Forest

# Load banknote dataset
# Available at http://archive.ics.uci.edu/ml/machine-learning-databases/00267/
print('\nLoading Data Banknote Authentication Dataset...\n')
data = np.genfromtxt('data_banknote_authentication.csv', delimiter=',')
np.random.shuffle(data)

# Input and labels
x, y = data[:, :-1], data[:, -1]
split = int(0.75 * x.shape[0])  # 75 % to train split

train_x, train_y = x[0:split], y[0:split]
validation_x, validation_y = x[split:-1], y[split:-1]

forest = Forest(max_depth=10, no_trees=3, min_samples_split=2,
                min_samples_leaf=1, feature_search=2, bootstrap=True)

forest.train(train_x, train_y)

print('\nMetrics :\n')
print("train error      : {:7.4f} %"
      "\nvalidation error : {:7.4f} %"
      "\nnumber of nodes  : {:7}  "
    .format(
    forest.eval(train_x, train_y),
    forest.eval(validation_x, validation_y),
    forest.node_count()
))
