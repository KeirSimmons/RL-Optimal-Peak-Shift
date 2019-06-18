from sim.scripts.run import Run
from sim.agents.DQN import DQN
from sim.envs.discrete_battery import DiscreteBattery
from sim.agents.PPO import PPO
from sim.envs.continuous_battery import ContinuousBattery

import argparse
import numpy as np
import random
import json
from collections import defaultdict

"""
Trains and tests an infinite number of models with each
model improving on the previous via automatic penalty shaping

Run from root:
python -m sim.train --agent x
python -m sim.train --agent x
(where x is either PPO or DQN - the algorithm to use)

UPDATED NOTE FOR THIS REPO: results directory and 
run commands will not work in this state (as the 
directory structure for this repository does not 
match the original).
"""

class Train:

    def __init__(self, households):
        self.households = households
        self.find_agent()
        self.train_models()

    def find_agent(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--agent", "-a", help="Choose from PPO or DQN", required=True, choices=['DQN', 'PPO']) 
        self.agent = eval(self.parser.parse_args().agent)

    def train_models(self):

        self.runner = None

        while True:
            
            # Choose random values for each parameter below
            self.generate_params()

            # Adjust coefficients of reward signals
            self.automatic_penalty_shaping()

            # Train & test a single model with these parameters
            self.train_model()

    def automatic_penalty_shaping(self):
        """
        The following uses automatic penalty shaping to adjust
        coefficients of reward signals (multipliers) based on
        previous model's results
        """

        if self.runner is not None:
            # Get penalties from previous model
            with open(self.runner.results.get_path(".penalties"), "r") as path:
                penalties_full = json.load(path)
                penalties = defaultdict(float)
                max_penalty = -np.inf
                for penalty in penalties_full:
                    penalties[penalty] = np.mean(penalties_full[penalty])
                    max_penalty = np.maximum(max_penalty, penalties[penalty])

            # Get coefficients from previous model
            with open(self.runner.results.get_path(".config"), "r") as path:
                config = json.load(path)
                multipliers = config['env_params']['multipliers']

            # Adjust new coefficients to force penalty averages to equal values
            new_multipliers = defaultdict(float)
            max_new_multiplier = -np.inf
            for penalty in penalties:
                new_multipliers[penalty] = self.multipliers[penalty] * max_penalty / penalties[penalty]
                max_new_multiplier = np.maximum(max_new_multiplier, new_multipliers[penalty])
            for penalty in penalties:
                new_multipliers[penalty] /= max_new_multiplier
            
            self.multipliers = new_multipliers        

    def generate_params(self):

        # Multiplicative constants for penalties
        multipliers = [
            "underpurchase",
            "abovelimit",
            "abovecapacity",
            "belowlimit",
            "cost"
        ]
        self.multipliers = {key:np.random.uniform() for key in multipliers}
        
        if self.agent.name == "PPO":
            self.env = ContinuousBattery
            self.agent_params = {
                'episode_length' : 1440
            }

        elif self.agent.name == "DQN":

            self.env = DiscreteBattery

            if np.random.uniform() <= 0.5:
                self.agent_params = {
                    'episodes' : 500,
                    'episode_length' : 1440
                }
            else:
                self.agent_params = {
                    'episodes' : 75,
                    'episode_length' : 10080
                }

            models = [
                [64],
                [64, 64],
                [64, 128, 64]
            ]
            self.agent_params['model_architecture'] = random.choice(models)

    def train_model(self):

        self.runner = Run(
            env=self.env,
            env_params={
                'households' : self.households,
                "multipliers" : self.multipliers
            },
            agent=self.agent,
            agent_params=self.agent_params
        )
        
        # Invoke train/test loop
        self.runner.full_cycle()

if __name__ == "__main__":
    # Pass a list of households to be trained on
    Train([
        "0002_9100000042"
    ])