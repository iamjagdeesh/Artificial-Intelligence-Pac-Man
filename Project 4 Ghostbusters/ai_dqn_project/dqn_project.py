from __future__ import print_function
import gym
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
from itertools import count
from replay_memory import ReplayMemory, Transition
import env_wrappers
import random
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--eval', action="store_true", default=False, help='Run in eval mode')
parser.add_argument('--seed', type=int, default=1, help='Random seed')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

class DQN(object):
    """
    A starter class to implement the Deep Q Network algorithm

    TODOs specify the main areas where logic needs to be added.

    If you get an error a Box2D error using the pip version try installing from source:
    > git clone https://github.com/pybox2d/pybox2d
    > pip install -e .

    """

    def __init__(self, env):

        self.env = env
        self.sess = tf.Session()

        # A few starter hyperparameters
        self.batch_size = 128
        self.gamma = 0.99
        # If using e-greedy exploration
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000  # in episodes
        # If using a target network
        self.clone_steps = 5000

        # memory
        self.replay_memory = ReplayMemory(100000)
        # Perhaps you want to have some samples in the memory before starting to train?
        self.min_replay_size = 10000

        # define yours training operations here...
        self.observation_input = tf.placeholder(tf.float32, shape=[None] + list(self.env.observation_space.shape))
        self.q_value = self.build_model(self.observation_input)

        # define your update operations here...
        self.action_input = tf.placeholder(tf.float32, [None, env.action_space.n])
        self.target_q_val = tf.placeholder(tf.float32, [None])
        self.update_op = self.update()

        self.num_episodes = 0
        self.num_steps = 0

        self.saver = tf.train.Saver(tf.trainable_variables())
        self.sess.run(tf.global_variables_initializer())

    def build_model(self, observation_input, scope='train'):
        """
        TODO: Define the tensorflow model

        Hint: You will need to define and input placeholder and output Q-values

        Currently returns an op that gives all zeros.
        """
        with tf.variable_scope(scope):
            x = layers.fully_connected(observation_input, 64, activation_fn=tf.nn.relu)
            x = layers.fully_connected(x, 32, activation_fn=tf.nn.relu)
            q_vals = layers.fully_connected(x, env.action_space.n, activation_fn=None)

            return q_vals

    def select_action(self, obs, evaluation_mode=False):
        """
        TODO: Select an action given an observation using your model. This
        should include any exploration strategy you wish to implement

        If evaluation_mode=True, then this function should behave as if training is
        finished. This may be reducing exploration, etc.

        Currently returns a random action.
        """
        obs = np.reshape(obs, [1, self.env.observation_space.shape[0]])
        if evaluation_mode:
            return np.argmax(self.sess.run(self.q_value, feed_dict={self.observation_input: obs}))
        else:
            if np.random.random() < self.eps_start:
                return env.action_space.sample()
            else:
                return np.argmax(self.sess.run(self.q_value, feed_dict={self.observation_input: obs}))

    def update(self):
        """
        TODO: Implement the functionality to update the network according to the
        Q-learning rule
        """
        action_q_val = tf.reduce_sum(tf.multiply(self.q_value, self.action_input), reduction_indices=1)
        # q_val_error = tf.reduce_mean(tf.squared_difference(self.target_q_val, action_q_val))
        q_val_error = tf.losses.huber_loss(self.target_q_val, action_q_val)
        # update_op = tf.train.RMSPropOptimizer(0.001).minimize(q_val_error)
        update_op = tf.train.GradientDescentOptimizer(0.001).minimize(q_val_error)

        return update_op

    def train(self):
        """
        The training loop. This runs a single episode.

        TODO: Implement the following as desired:
            1. Storing transitions to the ReplayMemory
            2. Updating the network at some frequency
            3. Backing up the current parameters to a reference, target network
        """
        done = False
        obs = env.reset()
        while not done:
            action = self.select_action(obs, evaluation_mode=False)
            next_obs, reward, done, info = env.step(action)
            self.replay_memory.push(obs, action, next_obs, reward, done)
            obs = next_obs

            self.num_steps += 1

            if self.num_steps % self.batch_size == 0:
                samples = self.replay_memory.sample(self.batch_size)
                for sample in samples:
                    sample_next_state_obs = np.reshape(sample[2], [1, self.env.observation_space.shape[0]])
                    target_val = sample[3] if sample[4] else sample[3] + self.gamma * np.max(
                        self.sess.run(self.q_value, feed_dict={self.observation_input: sample_next_state_obs}))

                    sample_cur_state_obs = np.reshape(sample[0], [1, self.env.observation_space.shape[0]])
                    action_ip = np.zeros(4)
                    action_ip[sample[1]] = 1
                    action_ip = np.reshape(action_ip, [1, 4])
                    self.sess.run(self.update_op,
                                  feed_dict={self.observation_input: sample_cur_state_obs, self.target_q_val: [target_val],
                                             self.action_input: action_ip})

        self.num_episodes += 1

        if self.num_episodes % self.eps_decay == 0 and self.eps_start > self.eps_end:
            self.eps_start -= self.eps_end

    def eval(self, save_snapshot=True):
        """
        Run an evaluation episode, this will call
        """
        total_reward = 0.0
        ep_steps = 0
        done = False
        obs = env.reset()
        while not done:
            env.render()
            action = self.select_action(obs, evaluation_mode=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
        # print ("Evaluation episode: ", total_reward)
        print("Evaluation episode: " + str(total_reward) + " Epsilon: " + str(self.eps_start) + " Episodes: " + str(self.num_episodes) + " Steps: " + str(self.num_steps))
        if save_snapshot:
            print ("Saving state with Saver")
            self.saver.save(self.sess, 'models/dqn-model', global_step=self.num_episodes)


def train(dqn):
    for i in count(1):
        dqn.train()
        # every 10 episodes run an evaluation episode
        if i % 10 == 0:
            dqn.eval()


def eval(dqn):
    """
    Load the latest model and run a test episode
    """
    ckpt_file = os.path.join(os.path.dirname(__file__), 'models/checkpoint')
    with open(ckpt_file, 'r') as f:
        first_line = f.readline()
        model_name = first_line.split()[-1].strip("\"")
    dqn.saver.restore(dqn.sess, os.path.join(os.path.dirname(__file__), 'models/'+model_name))
    dqn.eval(save_snapshot=False)


if __name__ == '__main__':
    # On the LunarLander-v2 env a near-optimal score is some where around 250.
    # Your agent should be able to get to a score >0 fairly quickly at which point
    # it may simply be hitting the ground too hard or a bit jerky. Getting to ~250
    # may require some fine tuning.
    env = gym.make('LunarLander-v2')
    env.seed(args.seed)
    # Consider using this for the challenge portion
    # env = env_wrappers.wrap_env(env)

    dqn = DQN(env)
    if args.eval:
        eval(dqn)
    else:
        train(dqn)
