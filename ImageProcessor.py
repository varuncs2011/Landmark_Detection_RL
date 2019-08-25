from rl.core import Processor


class ImageProcessor(Processor):
    """
    Required to transform an observation (which is a coordinate pair for the agent location) into
    the image patch that the agent observes.
    """

    def __init__(self, environment):
        self.env = environment

    def process_observation(self, observation):
        return self.env.get_window(observation)

    def process_reward(self, reward):
        return reward
