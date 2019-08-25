import numpy as np
from ActionSpace import *
from ObservationSpace import *




class ImageEnvironment:

    def dist(self, point):
        """Distance of point to the landmark"""
        return np.sqrt((point[0] - self.landmark[0]) ** 2 +
                       (point[1] - self.landmark[1]) ** 2)

    def get_window(self, observation_loc):
        """Retrieve the image patch representing the agent's view."""
        c_row, c_col = observation_loc

        state = self.image[
                c_row - self.offset: c_row + self.offset + 1,
                c_col - self.offset: c_col + self.offset + 1]

        return state

    def random_pos(self):
        pos = (self.rng.randint(self.offset + self.step_size, self.im_height - self.offset - self.step_size - 1),
               self.rng.randint(self.offset + self.step_size, self.im_width - self.offset - self.step_size - 1))

        return pos

    def __init__(self, images, landmarks, state_size, starting_pos=None, seed=None, step_size = 1):

        self.totalPass = 0
        self.totalFail = 0
        self.wander = 0
        self.maxEpisodeReached = 0
        self.terminal_points = list()
        self.images = images
        self.landmarks = landmarks
        self.rng = random.Random(seed)

        idx = self.rng.randint(0, len(self.landmarks) - 1)
        self.image = self.images[idx]
        self.landmark = self.landmarks[idx]

        assert (state_size % 2 == 1)  # Easier if the state window is of odd shape.
        self.state_size = state_size
        self.im_height = self.image.shape[0]
        self.im_width = self.image.shape[1]
        self.offset = state_size // 2  # number of pixels in the state window from the border to the center.
        self.debug = False  # Prints each step if enabled.
        self.step_size = 1

        # The starting position can't be at the borders of the image.
        self.starting_pos = starting_pos if not starting_pos else self.random_pos()
        self.current_pos = self.starting_pos
        self.moves_made = 0

        self.action_space = ActionSpace()
        self.observation_space = ObservationSpace(self.im_height, self.im_width, self.offset)

        # For visualisation
        self.shown = False
        self.past_rs = []
        self.past_cs = []
        self.redraw = True

    def step(self, action):
        action = self.action_space.actions[action]
        assert self.action_space.contains(action), "No such action: " + str(action)

        r, c = self.current_pos
        self.past_rs.append(r)
        self.past_cs.append(c)

        current_distance = self.dist(self.current_pos)

        if action == "UP":
            self.current_pos = (r - self.step_size, c)
        elif action == "DOWN":
            self.current_pos = (r + self.step_size, c)
        elif action == "LEFT":
            self.current_pos = (r, c - self.step_size)
        elif action == "RIGHT":
            self.current_pos = (r, c + self.step_size)

        new_distance = self.dist(self.current_pos)

        self.terminal_points.append(self.current_pos)

        flag = False

        if len(self.terminal_points) > 20:
            temp = self.terminal_points[-19:]

            self.terminal_points = list()
            total = temp.count(self.current_pos)
            if total > 8:
                distance = self.dist(self.current_pos)
                if distance <= 5:
                    self.totalPass += 1
                    print("Pass", new_distance)
                else:
                    self.totalFail += 1
                    print("Fail", new_distance)
                flag = True

        reward = current_distance - new_distance

        observation = self.current_pos

        # Agent has hit the margin.
        done = (self.current_pos[0] <= self.offset + self.step_size or
                self.current_pos[1] <= self.offset + self.step_size or
                self.current_pos[0] >= self.im_height - self.offset - self.step_size or
                self.current_pos[1] >= self.im_width - self.offset - self.step_size)

        if new_distance < 7 and current_distance >= new_distance:
            reward = 5

        if current_distance < 3:
            if new_distance >= 3:
                reward = -10

        # agent has wander off the image
        if done:
            self.wander += 1
            reward = -20

        # if flag is true, we have found the landmark .
        if flag:
            done = flag
            if new_distance <= 5:
                reward = 5

        self.moves_made += 1

        # Need to cast information to strings to prevent the framework from
        # processing it for accumulation.
        info = {"landmark": str(self.landmark), "moves": str(self.moves_made)}

        if done:
            self.reset()
            # Set to False to not redraw the environment after it's been reset and
            # thus losing the previous path.
            self.redraw = False
        return observation, reward, done, info

    def reset(self):

        idx = self.rng.randint(0, len(self.landmarks) - 1)
        self.image = self.images[idx]
        self.landmark = self.landmarks[idx]

        self.starting_pos = self.random_pos()
        self.current_pos = self.starting_pos
        self.moves_made = 0
        self.past_rs = []
        self.past_cs = []
        self.shown = False

        return self.current_pos

    def render(self, mode):
        pass

    def close(self):
        pass
