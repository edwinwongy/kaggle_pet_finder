from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

import random

random.seed(1994)

girls = {
    'height': random.sample(range(150, 170), 20),
    'weight': random.sample(range(40, 65), 20)
}

boys = {
    'height': random.sample(range(160, 180), 20),
    'weight': random.sample(range(60, 80), 20)
}

total = {**girls, **boys}

data = {
    'height': [155, 160, 165]
}
# tree = DecisionTreeClassifier(random_state=0)
# tree.fit()
