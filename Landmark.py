from Logging import *
from ImageEnvironment import *
from ImageProcessor import *
from DQNModel import *
from ParseData import *
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from keras.optimizers import Adam


P = ParseData()
# train and test data frame
train_df, test_df = P.get_Data()

train_images, test_images, train_landmarks, test_landmarks = P.transform(train_df, test_df)

nb_actions = 4
state_window_size = 13
image_size = 50
start_eps = 1.0
end_eps = 0.05

learning_rate = 1e-5
gamma = 0.001
batch_size = 32
step_size = 1
memory_length = 10

# Set up the neural network model. The Permute layer is needed since due to the
# framework the image comes in with channel axis first.
D = DQNModel()

model = D.create_model(memory_length, state_window_size, nb_actions)
# model.summary()

# Window length represents how many frames the agent "remembers".
memory = SequentialMemory(limit=500, window_length=memory_length)

# Policy determines how much the agent explores. Since the rewards are not sparse
# in our case, not much exploration is technically needed. A purely greedy policy
# might work for the gradient images.

policy = GreedyQPolicy()
# Make a training environment.
train_env = ImageEnvironment(
    images=train_images,
    landmarks=train_landmarks,
    state_size=state_window_size,
    seed=1,
    step_size=step_size)

train_processor = ImageProcessor(train_env)

dqn = DQNAgent(
    enable_double_dqn=False,
    model=model,
    nb_actions=nb_actions,
    gamma=gamma,
    batch_size=batch_size,
    memory=memory,
    nb_steps_warmup=50000,
    target_model_update=1e2,
    policy=policy,
    processor=train_processor)


dqn.compile(Adam(lr=learning_rate), metrics=['accuracy'])
dqn.processor = train_processor

experiment_name = "NoseTip"


history_train = dqn.fit(train_env, nb_steps=500,
                        nb_max_episode_steps=100, log_interval=30000,  visualize=False, verbose=2)


dqn.save_weights(experiment_name, overwrite=True)

print("******", train_env.wander)

L = Logging()
episode_count = L.log_train_history(history_train, experiment_name)


test_env = ImageEnvironment(images=test_images, landmarks=test_landmarks, state_size=state_window_size, seed=6,step_size=step_size)
test_env.debug = False


dqn.load_weights(experiment_name)
# Need to update the agent to have the test environment processor
dqn.processor = ImageProcessor(test_env)

# Test on one of the episodes
nb_episodes = 300
history_test = dqn.test(test_env, nb_episodes=300, nb_max_episode_steps=100, visualize=False)


L.log_test_history(history_test, experiment_name)
L.log_Test_results(test_env.totalPass, test_env.totalFail, test_env.wander, nb_episodes, experiment_name)
