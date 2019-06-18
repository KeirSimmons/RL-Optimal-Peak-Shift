from sim.agents.interface import IRLUnit
from baselines import deepq

import numpy as np

"""
DQN agent (uses openai baselines implementation)
"""

class DQN(IRLUnit):

    name = "DQN"

    def __init__(self, results):
        self.results = results
        self._act = None
        self.sess = None

    def train(self,
              env,
              episodes=None,
              episode_length=None,
              model_architecture=None,
              lr=1e-3,
              max_timesteps=None,
              buffer_size=50000,
              exploration_fraction=0.15,
              exploration_final_eps=0.01,
              train_freq=1,
              batch_size=32,
              print_freq=1,
              checkpoint_freq=None,
              checkpoint_path=None,
              learning_starts=None,
              gamma=1.0,
              target_network_update_freq=500,
              prioritized_replay=True,
              prioritized_replay_alpha=0.6,
              prioritized_replay_beta0=0.4,
              prioritized_replay_beta_iters=None,
              prioritized_replay_eps=1e-6,
              param_noise=False,
              callback=None):

        # Ensure required params are passed
        # episodes is required for max_timesteps
        # Really episode_length is not required by an agent in general but we use it for max_timesteps and other vars
        if episodes is None or episode_length is None or model_architecture is None:
            raise ValueError("Required parameters not passed")
            
        self.model_architecture = model_architecture

        # Set default values for various agent parameters
        max_timesteps = max_timesteps if max_timesteps is not None else episodes * episode_length
        callback = callback if callback is not None else lambda x, y: False
        checkpoint_freq = checkpoint_freq if checkpoint_freq is not None else episode_length
        learning_starts = learning_starts if learning_starts is not None else episode_length

        # Agent parameters
        self.params = {
            'lr' : lr,
            'max_timesteps' : max_timesteps,
            'buffer_size' : buffer_size,
            'exploration_fraction' : exploration_fraction,
            'exploration_final_eps' : exploration_final_eps,
            'train_freq' : train_freq,
            'batch_size' : batch_size,
            'print_freq' : print_freq,
            'checkpoint_freq' : checkpoint_freq,
            'checkpoint_path' : checkpoint_path,
            'learning_starts' : learning_starts,
            'gamma' : gamma,
            'target_network_update_freq' : target_network_update_freq,
            'prioritized_replay' : prioritized_replay,
            'prioritized_replay_alpha' : prioritized_replay_alpha,
            'prioritized_replay_beta0' : prioritized_replay_beta0,
            'prioritized_replay_beta_iters' : prioritized_replay_beta_iters,
            'prioritized_replay_eps' : prioritized_replay_eps,
            'param_noise' : param_noise,
            'callback' : callback
        }

        self.model = deepq.models.mlp(self.model_architecture)
        
        self.act, self.sess = deepq.learn(
            env,
            self.model,
            **self.params
        )

        save_path = self.results.get_path("model.pkl")
        self.act.save(save_path)
        print("Model saved to {}".format(save_path))

        to_return = {
            "episode_rewards" : env.episode_rewards,
            "episode_errors" : env.episode_errors,
            "episode_penalties" : env.episode_penalties
        }

        return to_return

    def play(self,
             env=None,
             data=None,
             timesteps=None,
             indices=None,
             household=None):
        
        if self.act is None:
            self.act, self.sess = deepq.load(self.results.get_path("model.pkl"))

        requested = np.zeros((timesteps,))
        limits = np.zeros((timesteps,))
        cost = np.zeros((timesteps,))
        prices = np.zeros((timesteps,))
        charges = np.zeros((timesteps,))

        t = 0
        for episode in range(env.data.cutoff, env.data.data_info["episodes"]):
            _, done = env.reset(rand=False, episode=episode, household=household, carry_charge=True, train=False), False
            while not done:
                adjusted_data = env.adjusted_data
                _, _, done, info = env.step(action=self.act(adjusted_data[None])[0])
                cost[t] = info['cost']
                requested[t] = info['requested']
                limits[t] = info['demand']
                prices[t] = info['price']
                charges[t] = info['charge']
                t += 1

            print("Week {} finished testing ({})".format(episode, DQN.name))

        cost = np.sum(cost.reshape((-1, 60*24)).mean(axis=0))

        return cost, requested, limits, prices, charges