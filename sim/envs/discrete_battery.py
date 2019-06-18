from sim.envs.interface import Interface
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from sim.scripts.dataset_load import DataLoad

import numpy as np
import time
from collections import defaultdict
import matplotlib.pyplot as plt

"""
Simple battery definition which can handle a discrete
action space
"""

class DiscreteBattery(Interface):
    
    def __init__(self,
                 max_capacity=14.4, # kWh
                 power_rating=10,
                 actions=None,
                 foresight=0,
                 episode_length=1440,
                 households=None,
                 multipliers=None):

        self.max_capacity = max_capacity
        self.power_rating = power_rating
        self.actions = [0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04] if actions is None else actions
        self.action_space = Discrete(len(self.actions),)
        self.episode_length = episode_length
        self.charge = 0.0

        self.foresight = foresight
        self.households = households
        self.load_data()

        self.episode_rewards = [] # stores rewards for each episode
        self.episode_errors = defaultdict(list) # stores errors for each episode
        self.accumulative_errors = defaultdict(int)
        self.possible_errors = [
            "underpurchase",
            "abovelimit",
            "abovecapacity"
        ]
        self.episode_penalties = defaultdict(list)
        self.possible_penalties = [
            "underpurchase",
            "abovelimit",
            "abovecapacity",
            "belowlimit",
            "cost"
        ]

        # multipliers for penalties
        self.multipliers = multipliers if multipliers is not None else {}
        for penalty in self.possible_penalties:
            if penalty not in self.multipliers:
                self.multipliers[penalty] = 1.0

        self.total_bought = 0.
        self.total_cost = 0.

        self.k = 0
        self.min = dict()
        self.max = dict()

        self.actions_in_episode = []

    @property
    def name(self):
        return "Battery"

    def reset(self, rand=True, episode=0, household=0, carry_charge=False, train=True):

        self.t = 0
        self.done = False
        self.reward = 0.0
        self.episode_reward = 0.0
        self.info = None
        self.actions_in_episode = []

        self.errors = defaultdict(int)
        self.penalties = defaultdict(float)

        if not carry_charge:
            self.charge = 0.0
            
        self.data.load_episode(rand=rand, episode=episode, household=household, train=train)
        self.observe()

        return self.adjusted_data

    def observe(self):
        
        try:
            self.original_data, self.adjusted_data = self.data.next() # this gets the current timestep observatory data
            self.original_data = np.append(self.original_data, self.charge)
            std = self.max_capacity / np.sqrt(12) # std of uniform on [0, self.max_capacity]; sigma = (b-a)/sqrt(12)
            self.charge_adjusted = (self.charge - self.max_capacity/2) / std # mean of half max capacity (uniform)
            self.adjusted_data = np.append(self.adjusted_data, self.charge_adjusted) # TODO: Update running average/std of SoC
            
        except StopIteration: # episode complete
            self.done = True

            var = np.var(self.actions_in_episode)
            mean = np.mean(self.actions_in_episode)

            for error in self.errors:
                self.accumulative_errors[error] += self.errors[error]

            #### CONFIGURE PLOTS ####
            self.episode_rewards.append(self.episode_reward)
            for error in self.possible_errors:
                self.episode_errors[error].append(self.errors[error])

            for penalty in self.possible_penalties:
                self.episode_penalties[penalty].append(self.penalties[penalty])
            #### CONFIGURE PLOTS ####
            
        return self.adjusted_data, self.reward, self.done, self.info

    def _map_action_to_purchase(self, action):
        return self.actions[action]

    def _standardise_penalty(self, penalty, label):
        if label in self.min:
            self.min[label] = np.minimum(self.min[label], penalty)
            self.max[label] = np.maximum(self.max[label], penalty)
        else:
            self.min[label] = penalty
            self.max[label] = penalty
        if self.max[label] - self.min[label] == 0:
            return 0
        else:
            return (penalty - self.min[label]) / (self.max[label] - self.min[label])

    def step(self, action=None, energy=None):

        self.k += 1
        self.reward = 0

        if energy is not None:
            to_buy = np.maximum(0, energy) # allows variable amount (not discretised)
        else: # otherwise use the passed action
            to_buy = self._map_action_to_purchase(action)

        self.actions_in_episode.append(to_buy)
        
        # The data we have is essentially for the next timestep (i.e. we look into the future)
        price_index = self.data.data_info['features']['price']['start']
        demand_index = self.data.data_info['features']['demand']['start']
        generation_index = self.data.data_info['features']['generation']['start']
        consumption_index = self.data.data_info['features']['consumption']['start']

        price = self.original_data[price_index]
        limit = self.original_data[demand_index]
        generated = self.original_data[generation_index]
        consumed = self.original_data[consumption_index]

        current_energy = self.charge + generated - consumed
        minimum_energy_required = np.maximum(0, -current_energy)
        
        # Not bought enough energy for consumption requirements
        if to_buy < minimum_energy_required:
            underpurchase = minimum_energy_required - to_buy # how much more energy we needed
            penalty = self._standardise_penalty(underpurchase, "underpurchase") * self.multipliers["underpurchase"]
            self.reward -= penalty
            self.penalties["underpurchase"] += penalty
            self.errors['underpurchase'] += 1

            # provide the additional extra energy
            ## WARNING: Agent might learn to just rely on this (hence the penalty)
            to_buy = minimum_energy_required

        # Bought more energy than is currently available
        if to_buy > limit:
            abovelimit = to_buy - limit
            penalty = self._standardise_penalty(abovelimit, "abovelimit") * self.multipliers["abovelimit"]
            self.reward -= penalty
            self.penalties["abovelimit"] += penalty
            self.errors["abovelimit"] += 1

        # Battery can't hold this much excess!
        if current_energy + to_buy > self.max_capacity:
            abovecapacity = current_energy + to_buy - self.max_capacity
            penalty = self._standardise_penalty(abovecapacity, "abovecapacity") * self.multipliers["abovecapacity"]

            # reduce the amount we buy to the maximum possible
            # we will still be above the min energy required
            # (if we weren't, we would be using some battery charge!)
            to_buy = self.max_capacity - current_energy
            self.reward -= penalty
            self.penalties["abovecapacity"] += penalty
            self.errors["abovecapacity"] += 1

        # Let's reward the agent based on how much under the demand they are!
        if to_buy < limit:
            percentage = 1 - (to_buy / limit) # 1 is highest, 0 lowest
            reward = self._standardise_penalty(percentage, "belowlimit") * self.multipliers["belowlimit"]
            self.reward += reward
            self.penalties["belowlimit"] += reward

        # It's possible that to_buy is negative (full charge and we generate > consume)
        # Just assume that the additional generated energy is dissipated
        if to_buy < 0:
            to_buy = 0

        cost = to_buy * price
        # We limit here, as even if to_buy == 0 we might be over due to generate > consume
        self.charge = np.minimum(current_energy + to_buy, self.max_capacity)
        self.total_bought += to_buy
        self.total_cost += cost

        # Let's also penalise the amount we are spending!
        # We want to do this on to_buy * price, but we should standardise
        # both of these first before multiplying
        price_penalty = self._standardise_penalty(price, "price")
        tobuy_penalty = self._standardise_penalty(to_buy, "to_buy")
        penalty = self._standardise_penalty(price_penalty * tobuy_penalty, "cost") * self.multipliers["cost"]
        self.reward -= penalty
        self.penalties["cost"] += penalty

        self.info = {
            'cost' : cost,
            'requested' : to_buy,
            'demand' : limit,
            'price' : price,
            'charge' : self.charge
        }

        #### CONFIGURE PLOTS ####
        self.episode_reward += self.reward
        #### CONFIGURE PLOTS ####

        self.t += 1

        return self.observe()

    def load_data(self):
        # TODO: Expose these and more through __init__ call
        self.data = DataLoad(
            episode_len=self.episode_length,
            households=self.households,
            foresight=self.foresight,
            log=True
        )

        low_space = np.array([*[-np.inf] * len(self.data.data_info['labels']), 0.0])
        high_space = np.array([*[np.inf] * len(self.data.data_info['labels']), self.max_capacity])

        self.observation_space = Box(low=low_space, high=high_space)
