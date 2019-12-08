"""Quick experiment to test various tree sizes on Sonar and Banknote datasets.

"""

from itertools import product

import pandas as pd
import numpy as np

from main import train_routine

SAMPLE = 10  # How many time to run each experiment
DATASETS = ['Banknote', 'Sonar']
NO_TREES = [1, 64, 128]

combinations = product(DATASETS, NO_TREES)

results = []

n = 0
for c in combinations:
    for _ in range(SAMPLE):
        n += 1
        print('\n\nExperiment no {}\n'.format(n))
        train_error, test_error, _ = train_routine(*c)
        results.append([*c, train_error, test_error])

df = pd.DataFrame(results, columns=['dataset', 'trees',
                                    'train_error', 'test_error'])

df['train_error'] = (df['train_error']).round(2)
df['test_error'] = (df['test_error']).round(2)

print('\n\nMinimum of all experiments :\n')
df2 = pd.pivot_table(df, values=['test_error'],
                     index='dataset', columns='trees', aggfunc=np.min)
print(df2)

print('\n\nAverage of all experiments :\n')
df3 = pd.pivot_table(df, values=['test_error'],
                     index='dataset', columns='trees', aggfunc=np.mean)
print(df3.round(2))

df.to_csv('results.csv', index=False)
