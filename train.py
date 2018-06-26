import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import datetime

from reinforce import Reinforce, policy_network
from net_manager import NetManager


def parse_args():
    desc = 'TensorFlow implementation of "Neural Architecture Search with Reinforcement Learning"'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--max_layers', default=2)
    args = parser.parse_args()
    args.max_layers = int(args.max_layers)
    return args


def train(mnist):
    global args
    sess = tf.Session()
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.99, global_step, 500, 0.96, staircase=True)
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

    reinforce = Reinforce(sess, optimizer, policy_network, args.max_layers, global_step)
    net_manager = NetManager(num_input=784, num_classes=10, learning_rate=0.001, mnist=mnist, batch_size=100)

    MAX_EPISODES = 2500
    step = 0
    state = np.array([[10.0, 128.0, 1.0, 1.0] * args.max_layers], dtype=np.float32)
    pre_acc = 0.0
    total_rewards = 0

    for i_episode in range(MAX_EPISODES):
        action = reinforce.get_action(state)
        print("action:", action)
        if all(ai > 0 for ai in action[0][0]):
            reward, pre_acc = net_manager.get_reward(action, step, pre_acc)
            print('====>', reward, pre_acc)
        else:
            reward = -1.0
        total_rewards += reward

        # In our sample action is equal state
        state = action[0]
        reinforce.store_rollout(state, reward)

        step += 1
        ls = reinforce.train_step(1)
        log_str = 'current time: ' + str(datetime.datetime.now().time()) + ' episode: ' + str(i_episode) + ' loss: ' + str(ls) + ' last_state: ' + str(state) + ' last_reward: ' + str(reward) + '\n'
        log = open('log.txt', 'a+')
        log.write(log_str)
        log.close()
        print(log_str)


def main(_):
    global args
    args = parse_args()
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()