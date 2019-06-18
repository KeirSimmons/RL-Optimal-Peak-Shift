from sim.scripts.run import Run
from sim.agents.PPO import PPO
from sim.agents.DQN import DQN
from sim.envs.discrete_battery import DiscreteBattery
from sim.envs.continuous_battery import ContinuousBattery

import argparse

"""
Allows multiple already trained models to be tested
Results are saved in ./results/

Run from root:
python -m sim.test -r x y z
(x, y, z are a list of result directory names)

UPDATED NOTE FOR THIS REPO: results directory and 
run commands will not work in this state (as the 
directory structure for this repository does not 
match the original).
"""

class Test:

    def __init__(self):
        self.find_dirs()
        self.run_all()

    def find_dirs(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--result_dir", "-r", nargs="+", help="The result directory number which needs to be played back (tested)", required=True)

    def run_all(self):
        for result_dir in self.parser.parse_args().result_dir:
            print("Playing back result_dir={}".format(result_dir))
            self.try_all(result_dir)

    def try_all(self, result_dir):
        try:
            self.launch_runner(ContinuousBattery, PPO, result_dir)
        except AssertionError:
            try:
                self.launch_runner(DiscreteBattery, DQN, result_dir)
            except AssertionError:
                print("Agent not found")

    def launch_runner(self, env, agent, result_dir):
        runner = Run(
            env=env,
            agent=agent,
            result_dir=result_dir
        )
        runner.initialise_agent()
        print("Using {} agent".format(str(agent)))
        runner.test()

if __name__ == "__main__":
    Test()