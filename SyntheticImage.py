import random
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import distance_transform_edt

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.core import Processor

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, Permute, SimpleRNN, LSTM, Reshape, MaxPool2D
from keras.optimizers import Adam


totalPass=0
totalFail=0
wander=0
maxEpisodeReached=0


def display_image(image):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='gray_r')
    ax.axis('off')  # clear x- and y-axes
    plt.show()


def make_image(seed, image_size=50):
    random.seed(seed)
    img = np.zeros((image_size, image_size)) + 1
    landmark = [random.randint(0, image_size - 1), random.randint(0, image_size - 1)]
    img[landmark[0], landmark[1]] = 0
    img = distance_transform_edt(img)  # Perform distance transform to the landmark
    img /= np.sqrt(2 * 50 ** 2)  # Normalise to the range of [0, 1]
    return img, landmark


def check_bounds(im, coord):
    if coord[0] >= 0 and coord[0] < im.shape[0] and coord[1] >= 0 and coord[1] < im.shape[1]:
        return True
    else:
        return False


# Function to draw a circle
def draw_circle(im, centre, radius, intensity):
    assert (len(centre) == 2)
    circle_min = centre[0] - radius, centre[1] - radius
    circle_max = centre[0] + radius, centre[1] + radius
    if (check_bounds(im, circle_min) == False): return None
    if (check_bounds(im, circle_max) == False): return None

    radius_squared = pow(radius, 2)
    for y in range(int(circle_min[0]), int(circle_max[0]) + 1):
        for x in range(int(circle_min[1]), int(circle_max[1]) + 1):
            if (pow(y - centre[0], 2) + pow((x - centre[1]), 2)) <= radius_squared:
                im[y, x] = intensity
    return im


# Function to draw a rectangle
def draw_rectangle(im, rect_min, rect_max, intensity):
    if (check_bounds(im, rect_min) == False): return None
    if (check_bounds(im, rect_max) == False): return None

    for y in range(int(rect_min[0]), int(rect_max[0]) + 1):
        for x in range(int(rect_min[1]), int(rect_max[1]) + 1):
            im[y, x] = intensity
    return im


# Generate a face image.
def make_face_image(seed, im_size=50):
    # Generate some random image appearance parameters.
    random.seed(seed)
    face_position = [20 + round(random.random() * 10), 20 + round(random.random() * 10)]
    face_radius = 14 + round(random.random() * 5)
    eye_height = face_position[0] - 2 - round(random.random() * 6)
    eye_radius = 1 + round(random.random() * 3)
    face_intensity = 10 + random.random() * 20
    eye_intensity = 64.0 + random.random() * 5

    centre_eye_R = [eye_height, face_position[1] - 6]
    centre_eye_L = [eye_height, face_position[1] + 6]
    vertex = [face_position[0] - face_radius, face_position[1]]
    mouth_L = [face_position[0] + 5, face_position[1] + 5]

    # Generate image.
    im = np.zeros((im_size, im_size)) + 100
    im = draw_circle(im, face_position, face_radius, face_intensity)
    im = draw_circle(im, centre_eye_R, eye_radius, eye_intensity)
    im = draw_circle(im, centre_eye_L, eye_radius, eye_intensity)
    im = draw_rectangle(im, [face_position[0] + 4, face_position[1] - 5], mouth_L, 50)
    im = im + 4 * np.random.randn(im_size, im_size)

    lmks = [centre_eye_R, centre_eye_L, vertex, mouth_L]

    # Only return left eye for now

    return im, lmks[0]


terminal_points = list()


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


class ImageEnv:

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

        if state.shape[0] != 7 or state.shape[1] != 7:
            print("state shape", state.shape)
            print(observation_loc)
            print("moves:" + str(self.moves_made))

        return state

    def random_pos(self):
        pos = (self.rng.randint(self.offset + self.step_size, self.im_height - self.offset - self.step_size - 1),
               self.rng.randint(self.offset + self.step_size, self.im_width - self.offset - self.step_size - 1))
        #print("random pos: " + str(pos))
        return pos

    def __init__(self, images, landmarks, state_size=7, starting_pos=None, seed=None, step_size=1):
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
        self.step_size = step_size

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

        if self.debug:
            print("Landmark: " + str(self.landmark))
            print("Current pos: " + str(self.current_pos))
            print("current distance: " + str(current_distance))

        if action == "UP":
            self.current_pos = (r - step_size, c)
        elif action == "DOWN":
            self.current_pos = (r + step_size, c)
        elif action == "LEFT":
            self.current_pos = (r, c - step_size)
        elif action == "RIGHT":
            self.current_pos = (r, c + step_size)

        new_distance = self.dist(self.current_pos)

        terminal_points.append(self.current_pos)

        flag = False
        if len(terminal_points) > 20:
            temp = terminal_points[-19:]
            total = temp.count(self.current_pos)
            if total > 8:
                distance = self.dist(self.current_pos)
                if distance <= 2:
                    global totalPass
                    totalPass += 1
                else:
                    global totalFail
                    totalFail += 1
                flag = True

        reward = current_distance - new_distance

        if self.debug:
            print("reward: " + str(reward))

        observation = self.current_pos

        # Agent has hit the margin.
        done = (self.current_pos[0] <= self.offset + self.step_size or
                self.current_pos[1] <= self.offset + self.step_size or
                self.current_pos[0] >= self.im_height - self.offset - self.step_size or
                self.current_pos[1] >= self.im_width - self.offset - self.step_size)

        # agent has wander off the image
        if done:
            global wander
            wander += 1

        # if flag is true, we have found the landmark .
        if flag:
            done = flag

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
        # Pick a different image
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


class ImageEnvProcessor(Processor):
    """
    Required to transform an observation (which is a coordinate pair for the agent location) into
    the image patch that the agent observes.
    """

    def __init__(self, environment):
        self.env = environment

    def process_observation(self, observation):
        return self.env.get_window(observation)


dataset_to_use = "faces" # "gradients"
if dataset_to_use == "gradients":
  train_images, train_landmarks = zip(*[make_image(seed=x) for x in range(5000)])
  test_images, test_landmarks = zip(*[make_image(seed=x) for x in range(5000, 6000)])
elif dataset_to_use == "faces":
  train_images, train_landmarks = zip(*[make_face_image(seed=x) for x in range(5000)])
  test_images, test_landmarks = zip(*[make_face_image(seed=x) for x in range(5000, 6000)])

# Training parameters
nb_actions = 4  # We have 4 actions
state_window_size = 7  # Alison: This should be 7x7 because of the 3 sets of 3x3 convolution.
image_size = 50
start_eps = 1.0
end_eps = 0.05

learning_rate = 0.00025
gamma = 0.01  # ALison: No discount because we want to pay attention to all steps.
batch_size = 32
step_size = 1
memory_length = 10

# Set up the neural network model. The Permute layer is needed since due to the
# framework the image comes in with channel axis first.
model = Sequential()
model.add(Permute((2, 3, 1), input_shape=(memory_length, state_window_size, state_window_size)))
model.add(Conv2D(30, 3, padding="valid"))
model.add(Activation("relu"))
model.add(Conv2D(60, 3, padding="valid"))
model.add(Activation("relu"))
model.add(Conv2D(120, 3, padding="valid"))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dense(50))
model.add(Activation("relu"))
model.add(Dense(nb_actions))  # predict rewards for every of the 4 actions
model.add(Activation("linear"))
model.summary()

# Window length represents how many frames the agent "remembers".
memory = SequentialMemory(limit=50000, window_length=memory_length)

# Policy determines how much the agent explores. Since the rewards are not sparse
# in our case, not much exploration is technically needed. A purely greedy policy
# might work for the gradient images.
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                              attr='eps',
                              value_max=start_eps,
                              value_min=end_eps,
                              value_test=.05,
                              nb_steps=200000)

# Make a training environment.
train_env = ImageEnv(
    images=train_images,
    landmarks=train_landmarks,
    state_size=state_window_size,
    seed=1,
    step_size=step_size)

train_processor = ImageEnvProcessor(train_env)

dqn = DQNAgent(
    enable_double_dqn=True,
    model=model,
    nb_actions=nb_actions,
    gamma=gamma,
    batch_size=batch_size,
    memory=memory,
    nb_steps_warmup=50000,  # ALNE: Adding same number of warm-up steps as there is space in memory.
    target_model_update=1e2,
    policy=policy,
    processor=train_processor)

dqn.compile(Adam(lr=learning_rate), metrics=['mae'])



dqn.processor = train_processor
history_train = dqn.fit(train_env, nb_steps=200000, nb_max_episode_steps=300, visualize=False, verbose=0)
experiment_name = "centre_eye_Rightwitherror"
dqn.save_weights(experiment_name, overwrite=True)

episode_count = 0
with open('Train_'+experiment_name+'.txt', 'w') as f:
    for key, val in history_train.history.items():
        episode_count = len(val)
        for s in val:
            f.write(key + ":" + str(s))
            f.write("\n")
        f.write("******")

print("Total Pass:"+str(totalPass))
print("Total Fail:"+str(totalFail))
print("Total Wander:"+str(wander))
print("Total MaxEpisodeReached:"+str(episode_count-totalPass-totalFail-wander))

with open('Results_Train'+experiment_name+'.txt', 'w') as f:
    f.write("Total Pass:"+str(totalPass))
    f.write("\n")
    f.write("Total Fail:" + str(totalFail))
    f.write("\n")
    f.write("Total Wander:" + str(wander))
    f.write("\n")
    f.write("Total MaxEpisodeReached:" + str(episode_count - totalPass - totalFail - wander))

totalPass=0
totalFail=0
wander=0
maxEpisodeReached = 0

test_env = ImageEnv(images=test_images, landmarks=test_landmarks, state_size=state_window_size, seed=6, step_size=step_size)
test_env.debug = False


#dqn.load_weights(experiment_name)
# Need to update the agent to have the test environment processor
dqn.processor = ImageEnvProcessor(test_env)

# Test on one of the episodes
nb_episodes = 1000
history_test = dqn.test(test_env, nb_episodes=1000, nb_max_episode_steps=100, visualize=False)

print("Total Pass:"+str(totalPass))
print("Total Fail:"+str(totalFail))
print("Total Wander:"+str(wander))
print("Total MaxEpisodeReached:"+str(nb_episodes-totalPass-totalFail-wander))


with open('Test_'+experiment_name+'.txt', 'w') as f:
    for key, val in history_test.history.items():
        for s in val:
            f.write(key + ":" + str(s))
            f.write("\n")
        f.write("******")


with open('Results_Test'+experiment_name+'.txt', 'w') as f:
    f.write("Total Pass:"+str(totalPass))
    f.write("\n")
    f.write("Total Fail:" + str(totalFail))
    f.write("\n")
    f.write("Total Wander:" + str(wander))
    f.write("\n")
    f.write("Total MaxEpisodeReached:" + str(nb_episodes - totalPass - totalFail - wander))


