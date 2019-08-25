
import random


class ActionSpace:
    """Represents the sample space of our 4 actions."""

    def __init__(self, seed=None):
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.rng = random.Random(seed)

    def sample(self):
        return self.rng.sample(self.actions)

    def seed(self, seed):
        self.rng.seed(seed)

    def contains(self, x):
        return x in self.actions
