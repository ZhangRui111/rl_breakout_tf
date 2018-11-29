import gym
import numpy as np
import os
import tensorflow as tf

from brain.dqn_2015 import DeepQNetwork
from network.network_dqn_2015 import build_network
from hyper_paras.hp_dqn_2015 import Hyperparameters

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def train_model(token, learning_rate=None, discount_factor=None):
    env = gym.make('Breakout-v0')
    # env = env.unwrapped
    # print(env.action_space)
    # print(env.observation_space)
    hp = Hyperparameters()

    if learning_rate is None:
        learning_rate = hp.LEARNING_RATE
    if discount_factor is None:
        discount_factor = hp.DISCOUNT_FACTOR

    bn = build_network(learning_rate)
    brain = DeepQNetwork(network_build=bn, hp=hp, token=token, discount_factor=discount_factor)

    total_steps = 0
    for i_episode in range(brain.hp.MAX_EPISODES):

        observation = env.reset()
        observation = brain.preprocess_image(observation)
        state = np.stack([observation] * 4)

        ep_reward = 0

        while True:
            # env.render()
            action = brain.choose_action(state)
            observation_, reward, done, info = env.step(action)
            observation_ = brain.preprocess_image(observation_)
            next_state = np.concatenate([state[1:], np.expand_dims(observation, 0)], axis=0)

            brain.store_transition(state, action, reward, next_state)

            if total_steps > brain.replay_start:
                brain.learn()

            ep_reward += reward

            if not done:
                state = next_state
            else:
                print('episode: ', i_episode, ' | reward: ', ep_reward)
                break

            observation = observation_
            total_steps += 1


def main():
    for learning_rate in [1E-4, 1E-3, 1E-5]:
        for discount_factor in [0.99, 0.25, 0.5, 0.75]:
            tf.reset_default_graph()
            token = str(learning_rate) + str(discount_factor)
            train_model(token, learning_rate, discount_factor)


if __name__ == '__main__':
    main()
