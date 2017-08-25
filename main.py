import random
from collections import deque

import gym
import numpy as np
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

BATCH_SIZE = 32
OBSERVATION_SPACE = 4
EPSILON_DECAY = .99
GAMMA = .9

epsilon = 1.
D = deque(maxlen=5000)
best = 0


def get_model():
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(4, 4)))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(2, activation='linear'))
    adam = Adam()
    model.compile(loss='mse', optimizer=adam)
    return model


def save_exp_replay(s0, a, r, s1, t):
    """
    :param s0 initial state
    :param a action taken
    :param r reward
    :param s1 next state
    :param t terminated?
    """
    D.append((s0, a, r, s1, t))


def get_exp_replay():
    try:
        return random.sample(D, BATCH_SIZE)
    except ValueError:
        return None


def discount_epsilon():
    global epsilon
    epsilon *= EPSILON_DECAY


def choose_action(s, model):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        s = np.expand_dims(s, axis=0)
        return np.argmax(model.predict(s)[0])


if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n

    model = get_model()

    while True:
        s = env.reset()
        i = 0
        # fill transitions array with the same state
        s = np.stack((s, s, s, s))
        terminated = False
        while not terminated:
            a = choose_action(s, model)
            discount_epsilon()
            s_prime, r, terminated, _ = env.step(a)

            # add the new frame to the transition array
            s_prime = np.vstack((s[1:], s_prime))
            save_exp_replay(s, a, r, s_prime, terminated)
            s = s_prime
            i += 1

        if i > best:
            best = i
            print("Best:", best)
        # train the network
        batch = get_exp_replay()  # get BATCH_SIZE samples
        if batch is None:
            continue

        X = np.zeros((len(batch), 4, 4))
        y = np.zeros((len(batch), 2))
        for idx, (ss, aa, rr, ss_prime, terminated) in enumerate(batch):
            X[idx] = ss
            ss = np.expand_dims(ss, axis=0)
            ss_prime = np.expand_dims(ss_prime, axis=0)
            Q_sa = model.predict(ss)[0]
            Q_sa_prime = model.predict(ss_prime)[0]

            if terminated:
                tt = r
            else:
                tt = r + GAMMA * np.max(Q_sa_prime[0])

            Q_sa[aa] = tt
            y[idx] = np.reshape(Q_sa, 2)

            model.fit(ss, np.reshape(Q_sa, (1, 2)), verbose=0, epochs=1)
