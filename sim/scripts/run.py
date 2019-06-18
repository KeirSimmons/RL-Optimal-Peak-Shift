from sim.scripts.results import Results
from sim.scripts.dataset_load import DataLoad
from sim.agents.baseline import Baseline

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
from collections import defaultdict
from scipy.optimize import curve_fit

"""
Handles the actual training and testing of models including
results and plot generation
"""

class Run:

    def __init__(self,
                 env=None,
                 env_params=None,
                 agent=None,
                 agent_params=None,
                 result_dir=None):

        # Setup figure counter
        self.figures = 0
        
        if result_dir is not None:
            # Load old results directory
            self.results = Results(result_dir=result_dir)

            # Get config file
            with open(self.results.get_path(".config"), "r") as path:
                json_obj = json.load(path)
                self.env_params = json_obj['env_params']
                self.agent_params = json_obj['agent_params']
                assert json_obj['agent'] == str(agent), "Agent does not match that used for training"
                assert json_obj['env'] == str(env), "Environment does not match that used during training"
                self.agent = agent
                self.env = env
            
        else:
            # Store environment and params
            self.env = env
            self.env_params = env_params

            # Handle results storage (directory creation etc)
            self.results = Results()

            # Create directories for individual household results
            self.household_dirs()            

            # Store parameters for the agent and initialise
            self.agent = agent
            self.agent_params = agent_params
            self.initialise_agent()

            # Log configuration setup (stores as .config in results directory)
            self.log_config()

    def new_figure(self):
        self.figures += 1
        plt.figure(self.figures)
        return self.figures

    def initialise_env(self):
        self._env = self.env(
            **self.env_params,
            episode_length=self.agent_params['episode_length']
        )

    def initialise_agent(self):
        self._agent = self.agent(self.results)

    def household_dirs(self):
        for hid in self.env_params['households']:
            path = self.results.get_path(hid)
            if not os.path.exists(path):
                os.makedirs(path)

    def log_config(self):

        # Note that this does not cover default params from agent or env classes

        full_config = {
            "env" : str(self.env),
            "agent" : str(self.agent),
            "env_params" : self.env_params,
            "agent_params" : self.agent_params
        }

        with open(self.results.get_path(".config"), "w") as config_file:
            config_file.write(json.dumps(full_config))

    def train(self):
        # Update model status to 2: currently training
        self.results.status(2)

        # Initialise environment
        self.initialise_env()

        # Trains the model
        self.train_data = self._agent.train(self._env, **self.agent_params)

        # Update model status to 3: fully trained
        self.results.status(3)

        # Save plots from training
        self.train_plots()

        # Let's get feedback on the penalties
        self.log_penalties()

        # Reset session
        self._agent.reset_session()

    def save_plot(self, xlabel, ylabel, fname, hid=None, legend=False, title=None, figure=None):
        if figure is not None:
            plt.figure(figure)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        if title is not None:
            plt.title(title)

        if legend:
            plt.legend()

        figure = plt.gcf()
        figure.set_size_inches(7, 3.7)

        if hid is not None:
            path = self.results.get_path(os.path.join(hid, "{}.png".format(fname)))
        else:
            path = self.results.get_path("{}.png".format(fname))
            
        plt.savefig(path, dpi=600)

        return figure

    def best_fit(self, x, a, b, c):
        return a * np.exp(-b * x) + c

    def train_plots(self):

        train_data = self.train_data

        # Plot reward curve
        self.new_figure()
        plt.plot(train_data['episode_rewards'], label='Rewards')
        self.save_plot('Episode [-]', 'Episode Reward [-]', 'episode_rewards', legend=True)
        try:
            xdata = np.asarray([float(x) for x in range(len(train_data['episode_rewards']))])
            popt, _ = curve_fit(self.best_fit, xdata, train_data['episode_rewards'])
            plt.plot(self.best_fit(xdata, *popt), label='Best Fit')
        except:
            pass

        self.save_plot('Episode [-]', 'Episode Reward [-]', 'episode_rewards_bestfit', legend=True)
        
        # Plot error curve
        self.new_figure()
        for key in train_data['episode_errors']:
            plt.plot(train_data['episode_errors'][key], label=key)
        self.save_plot("Episode [-]", "Errors [-]", 'errors', legend=True)

        # Plot penalties
        self.new_figure()
        for penalty in train_data['episode_penalties']:
            plt.plot(train_data['episode_penalties'][penalty], label=penalty)
        self.save_plot("Episode [-]", "Penalty [-]", 'penalties', legend=True)

    def play_plot(self, data, label, relative=False):
        data = np.asarray(data).reshape((-1, 60*24))
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        if relative:
            mean /= np.max(mean)
        plt.plot(mean, label=label)
        plt.fill_between([x for x in range(len(mean))], mean-std, mean+std, alpha=0.15)
        # offset times to start from midnight (depends on local timezone)
        for i in range(24):
            if datetime.datetime.fromtimestamp(i*3600).strftime('%H:%M') == '00:00': 
                start = i * 3600
                break
        ticks = 6
        x1 = [*np.arange(0, 60*24, 60*24/ticks), 60*24-1]
        x2 = [datetime.datetime.fromtimestamp(start+x*60).strftime('%H:%M') for x in x1]
        plt.xticks(x1, x2)

    def log_results(self, text, hid, overwrite=False):
        path = self.results.get_path(os.path.join(hid, "results.txt"))
        flag = "w" if overwrite else "a"
        with open(path, flag) as fhandle:
            fhandle.write(text + "\n")

    def log_penalties(self):
        penalties = {}
        for penalty in self.train_data['episode_penalties']:
            penalties[penalty] = list(self.train_data['episode_penalties'][penalty])
        path = self.results.get_path(".penalties")
        with open(path, "w") as fhandle:
            fhandle.write(json.dumps(penalties))

    def play(self, control=True, plots=None):

        prefix = "w/ control" if control else "baseline"

        # reinitialise environment
        self.initialise_env()
        
        # to get the indices of pricing data etc
        data_info = self.data.get_data(household=0)
        timesteps = self._env.data.data_info['timesteps_test']
        labels = data_info["household"]["labels"]

        indices = {
            "price" : labels.index("price_smart_life_plan"),
            "consumption" : labels.index("consumption_total"),
            "generation" : labels.index("generation_total"),
            "demand" : labels.index("demand_fixed")
        }

        if plots is None:
            plots = defaultdict(dict)

        def get_plot(hid, plot):
            if plot not in plots[hid]:
                plots[hid][plot] = self.new_figure()
            pid = plots[hid][plot]
            plt.figure(pid)
            return pid

        for household, hid in enumerate(self.env_params['households']):

            title = "Results with Battery" if control else "Baseline Results"
            title = "\n####################\n" + title + "\n####################\n"
            self.log_results(title, hid, overwrite=control)

            cost, requested, limits, prices, charge = self._agent.play(env=self._env,
                                                                       data=self.data, 
                                                                       timesteps=timesteps,
                                                                       indices=indices, 
                                                                       household=household)
            

            # Save all data from testing for use by the comparison tool
            if control:
                with open(self.results.get_path("{}/.data".format(hid)), "w") as path:
                    data = {
                        "requested" : list(requested),
                        "limits" : list(limits),
                        "price" : list(prices),
                        "charge" : list(charge)
                    }
                    json.dump(data, path)
            else:
                with open(self.results.get_path("{}/.baselinedata".format(hid)), "w") as path:
                    data = {
                        "requested" : list(requested),
                        "limits" : list(limits),
                        "price" : list(prices)
                    }
                    json.dump(data, path)

            for error in self._env.possible_errors:
                # For baseline we only care about the "abovelimit" error
                if control or error == "abovelimit":
                    amount = self._env.accumulative_errors[error]
                    percentage = self._env.accumulative_errors[error] / timesteps * 100
                    self.log_results("Error '{}': {} ({:.2f}%)".format(error, amount, percentage), hid)

            exceeded = np.sum(requested > limits)
            percentage_exceeded = exceeded / timesteps * 100
            self.log_results("Requested amount exceeds demand: {} times ({:.2f}%)".format(exceeded, percentage_exceeded), hid)

            self.log_results("Average daily bill = JPY {:.2f}".format(cost), hid)
            self.log_results("Average monthly bill = JPY {:.2f}".format(cost*365/12), hid)
            
            get_plot(hid, "requested")
            self.play_plot(requested, 'Requested Energy ({})'.format(prefix))
            if not control: # plot limit just once..
                self.play_plot(limits, 'Energy Limit')

            get_plot(hid, "requested_week")
            plt.plot(requested[:10080], label='Requested Energy ({})'.format(prefix))
            if not control: # plot limit just once..
                plt.plot(limits[:10800], label='Energy Limit')
                ticks = 6
                x1 = [*np.arange(0, 10080, 10080/ticks), 10079]
                x2 = ['Day {}'.format(c+1) for c, _ in enumerate(x1)]
                plt.xticks(x1, x2)

            get_plot(hid, "price")
            self.play_plot(requested, 'Requested Energy ({})'.format(prefix), relative=True)
            if not control: # only want to plot this once!
                self.play_plot(prices, 'Rel. Price of Energy', relative=True)
            
            print("Total energy bought = {:.2f} kWh".format(self._env.total_bought))
            self.log_results("Total energy bought = {:.2f} kWh".format(self._env.total_bought), hid)
            self.log_results("Total cost = JPY {:.2f}".format(self._env.total_cost), hid)
            self.log_results("Average price = JPY {:.2f} / kWh".format(self._env.total_cost/self._env.total_bought), hid)

            if charge is not None:
                get_plot(hid, "charge")
                self.play_plot(charge, 'Charge')

        return plots

    def test(self):

        self.data = DataLoad(
            episode_len=self.agent_params['episode_length'],
            households=self.env_params['households'],
            log = True
        )

        # Reset to force it to reload the model
        ## This is needed since we've exited the session
        self._agent.act = None
        agent_plots = self.play()

        # Reset session to allow a new model to trained
        self._agent.reset_session()

        self._agent = Baseline()
        baseline_plots = self.play(control=False, plots=agent_plots)

        for hid in baseline_plots:

            title = "Household {}".format(hid)

            self.save_plot("Time of Day [hh:mm]", "Requested Energy [kWh/min]", 'requested', legend=True, hid=hid, title=title, figure=baseline_plots[hid]['requested'])
            self.save_plot("Time of Day [hh:mm]", "Stored Energy [kWh/min]", 'requested_week', legend=True, hid=hid, title=title, figure=baseline_plots[hid]['requested_week'])
            self.save_plot("Time of Day [hh:mm]", "Normalised [-]", 'price', legend=True, hid=hid, title=title, figure=baseline_plots[hid]['price'])
            self.save_plot("Time of Day [hh:mm]", "Stored Energy [kWh/min]", 'charge', hid=hid, title=title, figure=baseline_plots[hid]['charge'])

        # Update model status to 4: finished
        self.results.status(4)
        plt.close('all')

    def full_cycle(self):
        self.train()
        self.test()