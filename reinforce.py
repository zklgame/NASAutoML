import tensorflow as tf
import numpy as np
import random


'''
    Policy network is a main network for searching optimal architecture
    it uses NAS - Neural Architecture Search recurrent network cell.
    https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/contrib/rnn/python/ops/rnn_cell.py#L1363

    Args:
        state: current state of required topology
        max_layers: maximum number of layers
    Returns:
        3-D tensor with new state (new topology)
'''
def policy_network(state, max_layers):
    with tf.name_scope('policy_network'):
        nas_cell = tf.contrib.rnn.NASCell(4 * max_layers)
        outputs, state = tf.nn.dynamic_rnn(
            nas_cell,
            tf.expand_dims(state, -1),
            dtype=tf.float32
        )
        bias = tf.Variable([0.05] * 4 * max_layers)
        outputs = tf.nn.bias_add(outputs, bias)
        return outputs[:, -1:, :]


class Reinforce():
    def __init__(self, sess, optimizer, policy_network, max_layers, global_step,
                 division_rate=100.0,
                 reg_param=0.001,
                 discount_factor=0.99,
                 exploration=0.0):
        self.sess = sess
        self.optimizer = optimizer
        self.policy_network = policy_network
        self.max_layers = max_layers
        self.global_step = global_step
        self.division_rate = division_rate
        self.reg_param = reg_param
        self.discount_factor = discount_factor
        self.exploration = exploration

        self.reward_buffer = []
        self.state_buffer = []

        self.create_variables()
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.sess.run(tf.variables_initializer(var_lists))

    def create_variables(self):
        with tf.name_scope('model_inputs'):
            # raw state representation
            self.states = tf.placeholder(tf.float32, [None, self.max_layers * 4], name='states')

        with tf.name_scope('predict_actions'):
            # initialize policy network
            with tf.variable_scope('policy_network'):
                self.policy_outputs = self.policy_network(self.states, self.max_layers)
            self.action_scores = tf.identity(self.policy_outputs, name='action_scores')
            self.predicted_action = tf.cast(tf.scalar_mul(self.division_rate, self.action_scores), tf.int32, name='predicted_action')

        # regularization loss
        policy_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='policy_network')
        # compute loss and gradients
        with tf.name_scope('compute_gradients'):
            # gradients for selecting action from policy network
            self.discounted_rewards = tf.placeholder(tf.float32, (None,), name='discounted_rewards')
            with tf.variable_scope('policy_network', reuse=True):
                self.logprobs = self.policy_network(self.states, self.max_layers)
                print("self.logprobs ", self.logprobs)

            # compute policy loss and regularization loss
            self.cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logprobs[:, -1, :], labels=self.states)
            self.pg_loss = tf.reduce_mean(self.cross_entropy_loss)
            self.reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_network_variables])
            self.loss = self.pg_loss + self.reg_param * self.reg_loss

            # compute gradients
            self.gradients = self.optimizer.compute_gradients(self.loss)
            # compute policy gradients
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    self.gradients[i] = (grad * self.discounted_rewards, var)

            # training update
            with tf.name_scope('train_policy_network'):
                # apply gradients to update policy network
                self.train_op = self.optimizer.apply_gradients(self.gradients, global_step=self.global_step)

    def get_action(self, state):
        # if random.random() < self.exploration:
        #     return np.array([[random.sample(range(1, 35), 4 * self.max_layers)]])
        # else:
        return self.sess.run(self.predicted_action, {self.states: state})

    def store_rollout(self, state, reward):
        self.state_buffer.append(state[0])
        self.reward_buffer.append(reward)

    def train_step(self, steps_count):
        states = np.array(self.state_buffer[-steps_count:]) / self.division_rate
        rewards = self.reward_buffer[-steps_count:]
        _, ls = self.sess.run([self.train_op, self.loss], {self.states: states, self.discounted_rewards: rewards})
        return ls
