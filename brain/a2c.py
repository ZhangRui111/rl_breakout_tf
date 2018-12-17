import cv2
import math
import numpy as np
import random

from brain.base_dqns import BaseDQN
from hyper_paras.hp_a2c import Hyperparameters


class Actor(object):
    def __init__(self, sess, network=None):
        self.sess = sess
        self.hp = Hyperparameters()

        self.state = network[0][0]
        self.action = network[0][1]
        self.td_error = network[0][2]
        self.acts_prob = network[1][0]
        self.exp_v = network[1][1]
        self.train_op = network[1][2]

    def learn(self, s, a, td):
        _, exp_v = self.sess.run([self.train_op, self.exp_v],
                                 feed_dict={self.state: s,
                                            self.action: a.reshape(-1),
                                            self.td_error: td.reshape(-1)})
        if math.isnan(exp_v) is True:
            print('nan for exp_v')
            raise Exception("nan error exp_v")

        return exp_v

    def choose_action(self, s):
        probs = self.sess.run(self.acts_prob, {self.state: s})  # get probabilities for all actions
        select_action = np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())
        return select_action  # return a int


class Critic(object):
    def __init__(self, sess, network=None):
        self.sess = sess
        self.hp = Hyperparameters()

        self.state = network[0][0]
        self.next_value = network[0][1]
        self.reward = network[0][2]
        self.value = network[1][0]
        self.td_error = network[1][1]
        self.loss = network[1][2]
        self.train_op = network[1][3]

    def learn(self, s, r, s_):
        v_ = self.sess.run(self.value, {self.state: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    feed_dict={self.state: s,
                                               self.next_value: v_.reshape(-1),
                                               self.reward: r.reshape(-1)})
        if np.sum(np.isnan(td_error)) >= 1:
            print('nan: {}'.format(np.sum(np.isnan(td_error))))
            raise Exception("nan error")

        return td_error


class A2C(BaseDQN):
    def __init__(self,
                 hp,
                 token,
                 network_actor,
                 network_critic,
                 prioritized=False,
                 initial_epsilon=None,
                 finial_epsilon=None,
                 finial_epsilon_frame=None,
                 discount_factor=None,
                 minibatch_size=None,
                 reply_start=None,
                 reply_memory_size=None,
                 target_network_update_frequency=None):
        super().__init__(hp,
                         token,
                         None,
                         prioritized,
                         initial_epsilon,
                         finial_epsilon,
                         finial_epsilon_frame,
                         discount_factor,
                         minibatch_size,
                         reply_start,
                         reply_memory_size,
                         target_network_update_frequency)
        self.network_actor = network_actor
        self.network_critic = network_critic
        self.actor = Actor(self.sess, network=self.network_actor)
        self.critic = Critic(self.sess, network=self.network_critic)

    def learn(self, incre_epsilon):
        self.learn_step_counter += 1

        # sample batch memory from all memory
        # zip(): Take iterable objects as parameters, wrap the corresponding elements in the object into tuples,
        # and then return a list of those tuples
        samples_batch = random.sample(self.memory, self.batch_size)  # list of tuples
        observation, eval_act_index, reward, observation_ = zip(*samples_batch)  # tuple of lists

        observation = np.array(observation)
        action = np.array(eval_act_index)
        reward = np.array(reward)
        observation_ = np.array(observation_)

        td_error = self.critic.learn(observation, reward, observation_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        self.actor.learn(observation, action, td_error)  # true_gradient = grad[logPi(s,a) * td_error]

    def preprocess_image(self, img):
        img = img[30:-15, 5:-5:, :]  # image cropping
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert from BGR to GRAY
        gray = cv2.resize(gray, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        return gray
