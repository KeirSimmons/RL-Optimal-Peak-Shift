from sim.scripts.run import Run
from sim.agents.PPO import PPO
from sim.agents.DQN import DQN
from sim.agents.baseline import Baseline
from sim.envs.discrete_battery import DiscreteBattery
from sim.envs.continuous_battery import ContinuousBattery

import os
import argparse
import json
import numpy as np
import datetime
import matplotlib.pyplot as plt

"""
Allows multiple already trained models to be compared against each other
Results are saved in ./results/comparisons/

Run from root:
python -m sim.compare -r x y z
(x, y, z are a list of result directory names)

UPDATED NOTE FOR THIS REPO: results directory and 
run commands will not work in this state (as the 
directory structure for this repository does not 
match the original).
"""

class Compare:

    def __init__(self):

        self.get_dirs()
        self.find_runners()
        self.find_households()
        self.get_data()
        self.generate_plots()

    def get_dirs(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--result_dir", "-r", nargs="+", help="The result directories to be compared against each other", required=True) 
        self.dirs = sorted(list(set(self.parser.parse_args().result_dir)))
        assert len(self.dirs) > 1, "You have to compare at least 2 results!"
        self.id = "-".join(self.dirs)
        self.path = os.path.join("results", "comparisons", self.id)
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def find_runners(self):
        self.runners = dict()

        for result_dir in self.dirs:
            print("Finding result_dir={}".format(result_dir))
            self.runners[str(result_dir)] = self.try_all(result_dir)

    def launch_runner(self, env, agent, result_dir):
        runner = Run(
            env=env,
            agent=agent,
            result_dir=result_dir
        )

        return runner

    def find_households(self):
        self.households = []
        for key in self.runners:
            runner = self.runners[key]
            if len(self.households) == 0:
                self.households = runner.env_params['households']
            else:
                # Ensure all models were trained on the same households
                assert set(self.households) == set(runner.env_params['households']), "Households in models do not match"

    def get_data(self):
        """
        This retrieves the saved data from the models to be plotted over each other
        """
        self.data = dict()
        self.baselinedata = dict()
        for hid in self.households:
            path = os.path.join(self.path, hid)
            if not os.path.exists(path):
              os.makedirs(path)
            self.data[hid] = dict()
            self.baselinedata[hid] = dict()
            for runner_count, key in enumerate(self.runners):
                runner = self.runners[key]
                with open(runner.results.get_path("{}/.data".format(hid)), "r") as path:
                    self.data[hid][key] = json.loads(path.read())
                if runner_count == 0:
                    with open(runner.results.get_path("{}/.baselinedata".format(hid)), "r") as path:
                        self.baselinedata[hid] = json.loads(path.read())                

    def generate_plots(self):
        
        self.plots = dict()
        self.figures = 0

        for hkey in self.data:
            for rkey in self.data[hkey]:
                # charge plot
                plot_id = "{}-{}".format(hkey, "charge")
                self.open_plot(plot_id, title="charge")
                data = self.data[hkey][rkey]["charge"]
                self.plot(data, rkey)

                # requested plot
                plot_id = "{}-{}".format(hkey, "requested")
                self.open_plot(plot_id, title="requested")
                data = self.data[hkey][rkey]["requested"]
                self.plot(data, "Requested Energy ({})".format(rkey))

                # price plot
                plot_id = "{}-{}".format(hkey, "price")
                self.open_plot(plot_id, title="price")
                data = self.data[hkey][rkey]["requested"]
                self.plot(data, "Requested Energy ({})".format(rkey), relative=True)

            # Add baseline data to requested plot
            plot_id = "{}-{}".format(hkey, "requested")
            self.open_plot(plot_id, title="requested")
            data = self.baselinedata[hkey]["requested"] # Actual requested
            self.plot(data, "Requested Energy (Baseline)")
            data = self.baselinedata[hkey]["limits"] # Limit
            self.plot(data, "Energy Limit")

            # Add baseline data to price plot
            plot_id = "{}-{}".format(hkey, "price")
            self.open_plot(plot_id, title="price")
            data = self.baselinedata[hkey]["requested"] # Actual requested
            self.plot(data, "Requested Energy (Baseline)", relative=True)
            data = self.baselinedata[hkey]["price"] # price
            self.plot(data, "Rel. Price of Energy", relative=True)
            
            self.save_plot("Time of Day [hh:mm]", "Requested Energy [kWh/min]", hkey, "{}-{}".format(hkey, "charge"))
            self.save_plot("Time of Day [hh:mm]", "Requested Energy [kWh/min]", hkey, "{}-{}".format(hkey, "requested"))
            self.save_plot("Time of Day [hh:mm]", "Normalised [-]", hkey, "{}-{}".format(hkey, "price"))

    def save_plot(self, xlabel, ylabel, hkey, plot_id):

        self.open_plot(plot_id)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # offset times to start from midnight (depends on local timezone)
        for i in range(24):
            if datetime.datetime.fromtimestamp(i*3600).strftime('%H:%M') == '00:00': 
                start = i * 3600
                break
                
        ticks = 6
        x1 = [*np.arange(0, 60*24, 60*24/ticks), 60*24-1]
        x2 = [datetime.datetime.fromtimestamp(start+x*60).strftime('%H:%M') for x in x1]
        plt.xticks(x1, x2)
        plt.legend()
        figure = plt.gcf()
        figure.set_size_inches(7, 3.7)
        plt.savefig(os.path.join(self.path, hkey, "{}.png".format(self.plots[plot_id]["title"])), dpi=600)

    def plot(self, data, label, relative=False):
        data = np.asarray(data).reshape((-1, 60*24))
        mean = data.mean(axis=0)
        if relative:
            mean /= np.max(mean)
        std = data.std(axis=0)
        plt.plot(mean, label=label)
        plt.fill_between([x for x in range(len(mean))], mean-std, mean+std, alpha=0.15)

    def open_plot(self, plot_id, title=None):

        if plot_id not in self.plots:
            self.figures += 1
            self.plots[plot_id] = {
                "figure" : self.figures,
                "title" : title
            }

        plt.figure(self.plots[plot_id]["figure"])

    def try_all(self, result_dir):
        try:
            return self.launch_runner(ContinuousBattery, PPO, result_dir)
        except AssertionError:
            try:
                return self.launch_runner(DiscreteBattery, DQN, result_dir)
            except AssertionError:
                print("Agent not found")

if __name__ == "__main__":
    Compare()