

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from IPython.display import clear_output
from time import sleep
import random


class Test:

    def __init__(self):
        self.rng = random.Random(1)

    def random_pos(self):
        pos = (self.rng.randint(6 + 1, 50 - 6 - 1 - 1),
               self.rng.randint(6 + 1, 50 - 6 - 1 - 1))

        return pos

t = Test()

#for i in range(100):

# print(t.random_pos())

train_data = pd.read_csv("training.csv",  sep=",")
test_data = pd.read_csv("training.csv",  sep=",")
print(train_data.shape)

print(train_data.head().T)

