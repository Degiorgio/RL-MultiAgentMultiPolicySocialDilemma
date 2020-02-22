import keras.backend.tensorflow_backend as backend
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Activation, Flatten
from keras.optimizers import Adam


def create_model(observation_space_values,
                 action_space_size,
                 learning_rate=0.001,
                 dropout=0.2):
    model = Sequential()

    # observation_space_values ex: (10, 10, 3) a 10x10 RGB image.
    model.add(Conv2D(256, (3, 3), input_shape=observation_space_values))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    # this converts our 3D feature maps to 1D feature vectors
    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(action_space_size, activation='linear'))
    model.compile(
        loss="mse",
        optimizer=Adam(lr=learning_rate), metrics=['accuracy']
    )
    return model
