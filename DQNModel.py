from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, Permute, MaxPool2D


class DQNModel:
    def __init__(self):
        pass

    def create_model(self, memory_length1, state_window_size1, nb_actions1):
        model = Sequential()
        model.add(Permute((2, 3, 1), input_shape=(memory_length1, state_window_size1, state_window_size1)))
        model.add(Conv2D(30, 3, padding="valid"))
        model.add(Activation("relu"))
        model.add(Conv2D(30, 3, padding="valid"))
        model.add(Activation("relu"))
        model.add(Conv2D(60, 3, padding="valid"))
        model.add(Activation("relu"))
        model.add(Conv2D(60, 3, padding="valid"))
        model.add(Activation("relu"))
        model.add(Conv2D(120, 3, padding="valid"))
        model.add(Activation("relu"))
        model.add(Conv2D(120, 3, padding="valid"))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(50))
        model.add(Activation("relu"))
        model.add(Dense(nb_actions1))  # predict rewards for every actions
        model.add(Activation("linear"))
        return model
