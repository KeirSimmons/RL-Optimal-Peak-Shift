from abc import ABCMeta, abstractmethod

"""
Interface/API for all environments (various types of storage systems,
for example). For now just used for BESS.
"""

class Interface:

    __metaclass__ = ABCMeta

    @property
    def name(self):
        return NotImplementedError

    """
    Return: the observation space
    """
    @abstractmethod
    def observation_space(self): raise NotImplementedError

    """
    Return: action space
    """
    @abstractmethod
    def action_space(self): raise NotImplementedError

    """
    Resets the environment
    Return: an observation (np array)
    """
    @abstractmethod
    def reset(self): raise NotImplementedError

    """
    Renders the environment
    Return: None
    """
    @abstractmethod
    def render(self): raise NotImplementedError

    """
    NB: reward and done are calculated via the _reward and _done methods to keep them separate
    Return: obs (np array), reward, done
    """
    @abstractmethod
    def observe(self): raise NotImplementedError

    """
    Calculates the reward based on the most recent observation
    Return: reward
    """
    @abstractmethod
    def get_reward(self, observation): raise NotImplementedError

    """
    Return: bool - True if episode is complete based on observation
    """
    @abstractmethod
    def is_done(self, observation): raise NotImplementedError

    """
    Takes an action in the environment
    Return: bool - True if action succeeded
    """
    @abstractmethod
    def step(self, action): raise NotImplementedError

    """
    Closes an environment (not always necessary to implement)
    Return: None
    """
    @abstractmethod
    def close(self): raise NotImplementedError

    """
    Visualises the agent playing in the environment
    model(obs) should return the action to take in the passed obs
    """
    def visualise(self, model, episodes=None):
        current_episode = 1
        while True:
            print("\n==================")
            print("Starting Episode {}".format(current_episode))
            print("==================\n")
            obs, done = self.reset(), False
            episode_rew = 0
            self.render()
            while not done:
                self.step(model(obs))
                obs, rew, done = self.observe()
                episode_rew += rew
                self.render()
            print("\nEpisode reward", episode_rew)
            current_episode += 1

            # Stop playing when we reach episode limit
            if episodes is not None and current_episode > episodes:
                break