import errno
import gym
import numpy as np
import os
import tensorflow as tf
import time

from brain.dqn_2015 import DeepQNetwork
from network.network_dqn_2015 import build_network
from hyper_paras.hp_dqn_2015 import Hyperparameters
from shared.utils import write_ndarray, read_ndarray, my_print

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def restore_parameters(sess, restore_path):
    """ Save and restore Network's weights.
    """
    saver = tf.train.Saver(max_to_keep=5)
    checkpoint = tf.train.get_checkpoint_state(restore_path)
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
        path_ = checkpoint.model_checkpoint_path
        step = int((path_.split('-'))[-1])
    else:
        # Re-train the network from zero.
        print("Could not find old network weights")
        step = 0
    return saver, step


def save_parameters(sess, save_path, saver, name):
    if not os.path.exists(os.path.dirname(save_path)):
        try:
            os.makedirs(os.path.dirname(save_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    saver.save(sess, name)
    my_print('save weights', '-')


def train_model(token, learning_rate=None, discount_factor=None):
    env = gym.make('Breakout-v0')
    # env = env.unwrapped
    # print(env.action_space)
    # print(env.observation_space)
    hp = Hyperparameters()
    data_list = np.arange(6).reshape((1, 6))

    if learning_rate is None:
        learning_rate = hp.LEARNING_RATE
    if discount_factor is None:
        discount_factor = hp.DISCOUNT_FACTOR

    bn = build_network(learning_rate)
    brain = DeepQNetwork(network_build=bn, hp=hp, token=token, discount_factor=discount_factor)

    saver, load_episode = restore_parameters(brain.sess, brain.graph_path)

    # train for numbers of episodes
    total_steps = 0
    for i_episode in range(brain.max_episode):

        observation = env.reset()
        observation = brain.preprocess_image(observation)
        state = np.stack([observation] * 4)

        ep_reward = 0
        num_step = 0
        start_time = time.time()

        while True:
            # env.render()
            action = brain.choose_action(state)
            observation_, reward, done, info = env.step(action)
            observation_ = brain.preprocess_image(observation_)
            next_state = np.concatenate([state[1:], np.expand_dims(observation_, 0)], axis=0)

            brain.store_transition(state, action, reward, next_state)

            ep_reward += reward
            num_step += 1

            if i_episode > brain.replay_start:
                brain.learn(done)

            if done:
                print('episode: ', i_episode, ' | reward: ', ep_reward, 'num_step: ', num_step, ' | total_step: ',
                      total_steps, 'epsilon', brain.epsilon, 'episode_time', time.time() - start_time)
                # save the log info.
                data_list = np.concatenate(
                    (data_list,
                     np.array([i_episode, ep_reward, num_step, total_steps, brain.epsilon, time.time() - start_time]).reshape((1, 6))))
                if data_list.shape[0] % brain.hp.OUTPUT_SAVER_ITER == 0:
                    write_ndarray(brain.graph_path + 'data', np.array(data_list))
                    data_list = data_list[-1, :].reshape(1, 6)
                if i_episode % brain.hp.WEIGHTS_SAVER_ITER == 0 and i_episode != 0:
                    save_parameters(brain.sess, brain.graph_path, saver, brain.graph_path + '-' + str(load_episode + i_episode))
                break

            state = next_state
            total_steps += 1


def main():
    for learning_rate in [1E-4, 1E-3, 1E-5]:
        for discount_factor in [0.99, 0.5]:
            tf.reset_default_graph()
            token = str(learning_rate) + str(discount_factor)
            train_model(token, learning_rate, discount_factor)


if __name__ == '__main__':
    main()
