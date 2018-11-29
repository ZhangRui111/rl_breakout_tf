import numpy as np
import random
import tensorflow as tf

from functools import reduce
from brain.base_dqns import BaseDQN
from shared.utils import my_print

# Clears the default graph stack and resets the global default graph.
# tf.reset_default_graph()


class DeepQNetwork(BaseDQN):
    def __init__(self,
                 network_build,
                 hp,
                 token,
                 initial_epsilon=None,
                 finial_epsilon=None,
                 finial_epsilon_frame=None,
                 discount_factor=None,
                 minibatch_size=None,
                 reply_start=None,
                 reply_memory_size=None,
                 target_network_update_frequency=None):
        super().__init__(network_build,
                         hp,
                         token,
                         initial_epsilon,
                         finial_epsilon,
                         finial_epsilon_frame,
                         discount_factor,
                         minibatch_size,
                         reply_start,
                         reply_memory_size,
                         target_network_update_frequency)

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0 and self.learn_step_counter != 0:
            self.sess.run(self.target_replace_op)
            my_print('target_params_replaced', '-')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        length = reduce(lambda x, y: x*y, self.n_features)
        observation = batch_memory[:, :length]
        eval_act_index = batch_memory[:, length].astype(int)
        reward = batch_memory[:, length + 1]
        observation_ = batch_memory[:, -length:]

        # input is all next observation
        q_eval_input_s_next, q_target_input_s_next = \
            self.sess.run([self.q_eval_net_out, self.q_target_net_out],
                          feed_dict={self.eval_net_input: observation_.reshape((-1, 210, 160, 3)),
                                     self.target_net_input: observation_.reshape((-1, 210, 160, 3))})
        # real q_eval, input is the current observation
        q_eval_input_s = self.sess.run(self.q_eval_net_out,
                                       {self.eval_net_input: observation.reshape((-1, 210, 160, 3))})
        if self.summary_flag:
            tf.summary.histogram("q_eval", q_eval_input_s)

        q_target = q_eval_input_s.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        selected_q_next = np.max(q_target_input_s_next, axis=1)

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self.train_op, self.loss],
                                     feed_dict={self.eval_net_input: observation.reshape((-1, 210, 160, 3)),
                                                self.q_target: q_target})
        # self.cost_his.append(self.cost)

        # epsilon-decay
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        if self.summary_flag:
            tf.summary.scalar("cost", self.cost)

        if self.summary_flag:
            # merge_all() must follow all tf.summary
            if self.flag:
                self.merge_op = tf.summary.merge_all()
                self.flag = False

        if self.summary_flag:
            merge_all = self.sess.run(self.merge_op,
                                      feed_dict={self.eval_net_input: observation.reshape((-1, 210, 160, 3)),
                                                 self.q_target: q_target})
            self.writer.add_summary(merge_all, self.learn_step_counter)
