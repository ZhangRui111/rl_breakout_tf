import gym
import numpy as np
import os
import tensorflow as tf
import time

from shared.utils import restore_parameters, save_parameters, write_ndarray, read_ndarray, my_print

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def train_model(brain, if_REINFORCE=False, if_a2c=False):
    env = gym.make('Breakout-v0')
    # env = env.unwrapped
    # print(env.action_space)
    # print(env.observation_space)
    if if_REINFORCE is True or if_a2c is True:
        data_list = np.arange(5).reshape((1, 5))
    else:
        data_list = np.arange(6).reshape((1, 6))

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
            if if_a2c is True or if_REINFORCE is True:
                action, probs = brain.choose_action(state)
            else:
                action = brain.choose_action(state)
            observation_, reward, done, info = env.step(action)
            observation_ = brain.preprocess_image(observation_)
            next_state = np.concatenate([state[1:], np.expand_dims(observation_, 0)], axis=0)

            brain.store_transition(state, action, reward, next_state)

            ep_reward += reward
            num_step += 1
            state = next_state
            total_steps += 1

            if if_REINFORCE is False and i_episode > brain.replay_start:
                brain.learn(done)

            if done:
                if if_REINFORCE is True or if_a2c is True:
                    if if_REINFORCE is True:
                        brain.learn()

                    print('episode: ', i_episode, ' | reward: ', ep_reward, 'num_step: ', num_step, ' | total_step: ',
                          total_steps, 'probs', probs, 'episode_time', time.time() - start_time)
                    # save the log info.
                    data_list = np.concatenate(
                        (data_list,
                         np.array([i_episode, ep_reward, num_step, total_steps, probs,
                                   time.time() - start_time]).reshape((1, 6))))

                if if_REINFORCE is False and if_a2c is False:
                    print('episode: ', i_episode, ' | reward: ', ep_reward, 'num_step: ', num_step, ' | total_step: ',
                          total_steps, 'epsilon', brain.epsilon, 'episode_time', time.time() - start_time)
                    # save the log info.
                    data_list = np.concatenate(
                        (data_list,
                         np.array([i_episode, ep_reward, num_step, total_steps, brain.epsilon,
                                   time.time() - start_time]).reshape((1, 6))))

                if data_list.shape[0] % brain.hp.OUTPUT_SAVER_ITER == 0:
                    write_ndarray(brain.graph_path + 'data', np.array(data_list))
                    data_list = data_list[-1, :].reshape(1, 6)
                if i_episode % brain.hp.WEIGHTS_SAVER_ITER == 0 and i_episode != 0:
                    save_parameters(brain.sess, brain.graph_path, saver,
                                    brain.graph_path + '-' + str(load_episode + i_episode))
                break


def main():
    # #parameters adjusting.
    # for learning_rate in [1E-4, 1E-3, 1E-5]:
    #     for discount_factor in [0.99, 0.5]:
    #         tf.reset_default_graph()
    #         token = str(learning_rate) + str(discount_factor)
    #         train_model(token, learning_rate, discount_factor)

    tf.reset_default_graph()
    # #choose model
    model = 'a2c'

    if model == 'double_dqn':
        print('double_dqn')
        from brain.double_dqn import DeepQNetwork
        from network.network_double_dqn import build_network
        from hyper_paras.hp_double_dqn import Hyperparameters

        hp = Hyperparameters()
        bn = build_network()
        token = 'double_dqn'  # token is useful when para-adjusting (tick different folder)
        brain = DeepQNetwork(hp=hp, token=token, network_build=bn)
        train_model(brain)
    elif model == 'dueling_dqn':
        print('dueling_dqn')
        from brain.dueling_dqn import DeepQNetwork
        from network.network_dueling_dqn import build_network
        from hyper_paras.hp_dueling_dqn import Hyperparameters

        hp = Hyperparameters()
        bn = build_network()
        token = 'dueling_dqn'  # token is useful when para-adjusting (tick different folder)
        brain = DeepQNetwork(hp=hp, token=token, network_build=bn)
        train_model(brain)
    elif model == 'pri_dqn':
        print('pri_dqn')
        from brain.pri_dqn import DeepQNetwork
        from network.network_pri_dqn import build_network
        from hyper_paras.hp_pri_dqn import Hyperparameters

        hp = Hyperparameters()
        bn = build_network()
        token = 'pri_dqn'  # token is useful when para-adjusting (tick different folder)
        brain = DeepQNetwork(hp=hp, token=token, network_build=bn)
        train_model(brain)
    elif model == 'REINFORCE':
        print('REINFORCE')
        from brain.REINFORCE import REINFORCE
        from network.network_REINFORCE import build_network
        from hyper_paras.hp_REINFORCE import Hyperparameters

        hp = Hyperparameters()
        bn = build_network()
        token = 'REINFORCE'  # token is useful when para-adjusting (tick different folder)
        brain = REINFORCE(hp=hp, token=token, network_build=bn)
        train_model(brain, if_REINFORCE=True)
    elif model == 'a2c':
        print('a2c')
        from brain.a2c import A2C
        from network.network_a2c import build_actor_network, build_critic_network
        from hyper_paras.hp_a2c import Hyperparameters

        hp = Hyperparameters()
        actor_bn = build_actor_network()
        critic_bn = build_critic_network()
        token = 'a2c'  # token is useful when para-adjusting
        brain = A2C(hp=hp, token=token, network_actor=actor_bn, network_critic=critic_bn)
        train_model(brain, if_a2c=True)
    else:
        print('No model satisfied, try dqn_2015!')
        from brain.dqn_2015 import DeepQNetwork
        from network.network_dqn_2015 import build_network
        from hyper_paras.hp_dqn_2015 import Hyperparameters

        hp = Hyperparameters()
        bn = build_network()
        token = 'dqn_2015'  # token is useful when para-adjusting
        brain = DeepQNetwork(hp=hp, token=token, network_build=bn)
        train_model(brain)


if __name__ == '__main__':
    main()
