from abc import ABCMeta, abstractmethod
import tensorflow as tf

"""
Defines the interface which all agents must inherit.
An agent connects the environment to a specific RL algorithm to provide the
ability to train an agent in the environment using said algorithm. Essentially
connects the 'wires' in the brain, but is not the brain itself.
"""

class IRLUnit:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def play(self, timestamp):
        raise NotImplementedError

    def reset_session(self):
        self.sess.__exit__(None, None, None)
        tf.reset_default_graph()