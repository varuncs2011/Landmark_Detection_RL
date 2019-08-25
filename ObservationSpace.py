import random


class ObservationSpace:
    """Observation space represents the possible fields of view for the agent."""

    def __init__(self, im_height, im_width, offset, seed=None):
        self.im_width = im_width
        self.im_height = im_height
        self.offset = offset
        self.rng = random.Random(seed)

    def sample(self):
        return (self.rng.randint(self.offset, self.im_height - self.offset - 1),
                self.rng.randint(self.offset, self.im_width - self.offset - 1))

    def seed(self, seed):
        self.rng.seed(seed)

    def contains(self, x):
        r, c = x
        return (self.offset < r < self.im_height - self.offset and
                self.offset < c < self.im_width - self.offset)
