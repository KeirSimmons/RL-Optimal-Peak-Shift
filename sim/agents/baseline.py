from sim.agents.interface import IRLUnit

import numpy as np

"""
Baseline agent (no control)
"""

class Baseline(IRLUnit):

    name = "Baseline"

    def __init__(self):
        pass

    def train(self, env):
        pass

    def reset_session(self):
        pass

    def play(self,
             env=None,
             data=None,
             timesteps=None,
             indices=None,
             household=None):

        requested = np.zeros((timesteps,))
        limits = np.zeros((timesteps,))
        cost = np.zeros((timesteps,))
        prices = np.zeros((timesteps,))

        t = 0
        for episode in range(env.data.cutoff, env.data.data_info["episodes"]):
            _, done = env.reset(rand=False, episode=episode, household=household, train=False), False
            while not done:
                original_data = env.original_data
                price = original_data[indices['price']]
                generated_energy = original_data[indices['generation']]
                consumed_energy = original_data[indices['consumption']]
                required_energy = np.maximum(0, consumed_energy + generated_energy - env.charge)
                demand = original_data[indices['demand']]
                _, _, done, info = env.step(energy=required_energy)
                cost[t] = required_energy * price
                requested[t] = required_energy
                limits[t] = demand
                prices[t] = price
                t += 1

            print("Week {} finished testing ({})".format(episode, Baseline.name))

        cost = np.sum(cost.reshape((-1, 60*24)).mean(axis=0))

        return cost, requested, limits, prices, None