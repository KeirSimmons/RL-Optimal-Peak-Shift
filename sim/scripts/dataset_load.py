import pandas as pd
import os
import shutil
import logging
from collections import defaultdict
import random
import numpy as np

"""
Handles all data loading and augmentation
UPDATED NOTE FOR THIS REPO: This of course,
will not do anything without the dataset
files, which are proprietary and will not
be distributed. 
"""

class DataLoad(object):

    def __init__(self,
                 path=None,
                 train_test=0.7, # train test split
                 episode_len=1440, # number of timesteps per ep (1 timestep = 1 min)
                 price_plan='smart_life_plan',
                 demand_plan='fixed',
                 households=None,
                 to_keep=None,
                 foresight=0,
                 forecast=None,
                 cutoff_episode_length=None, # if final ep has less than these timesteps, remove it (None = enforce full ep)
                 log=False):
        
        # choose whether or not to show INFO logs (mainly data)
        if log:
            logging.basicConfig(level=logging.INFO)

        # set path to data directory
        if path is None:
            # default path for data (relative to root energy_sim directory)
            path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data")

        self.path = path
        self.episode_len = episode_len
        self.households = households

        to_keep = {} if to_keep is None else to_keep
        to_keep['demand'] = [demand_plan] if 'demand' not in to_keep else to_keep['demand']
        to_keep['price'] = [price_plan] if 'price' not in to_keep else to_keep['price']
        to_keep['weather'] = [] if 'weather' not in to_keep else to_keep['weather']
        to_keep['consumption'] = ["total", "weekday"] if 'consumption' not in to_keep else to_keep['consumption']
        to_keep['generation'] = ["total"] if 'generation' not in to_keep else to_keep['generation']
        
        self.datasets = {
            "demand" : {
                "keep" : to_keep['demand']
            },
            "price" : {
                "keep" : to_keep['price']
            },
            "weather" : {
                "keep" : to_keep['weather']
            },
            "consumption" : {
                "list" : [
                    "dish_washer",
                    "washer",
                    "air_conditioner",
                    "ricecooker",
                    "ih",
                    "ecocute",
                    "refrigerator",
                    "tv",
                    "microwave",
                    "main",
                    "weekday"
                ],
                "keep" : to_keep['consumption'] # above plus 'other'; 'total' instead of 'main'
            },
            "generation" : {
                "list" : [
                    "photovoltaics"
                ],
                "keep" : to_keep['generation'] # above plus 'total'
            }
        }

        self.foresight = foresight # How many timesteps into the future the agent is allowed to see (for training)
        self.cutoff_episode_length = self.episode_len if cutoff_episode_length is None else cutoff_episode_length

        self.load_data() # Load in the datasets, and augment as appropriate
        self.cutoff = int(self.data_info["episodes"]*train_test)
        self.data_info['timesteps_test'] = (self.data_info["episodes"] - self.cutoff) * self.episode_len

    def load_data(self):
        # First check if augmented data already exists
        # order here is important! check_augmented_data() will create the directory tree structre so must come first!
        
        try:
            self.check_augmented_data()
        except Exception:
            self.augment_data()
        
        # This time if there's an error, we will allow the error to be raised and halt execution
        self.check_augmented_data()

        # let's get information on number of timesteps and features

        self.data_info = {
            "features" : defaultdict(dict),
            "timesteps" : 0,
            "episodes" : 0,
            "keys" : 0,
            "labels" : list()
        }

        for f1, ftype in enumerate(["original", "adjusted"]):
            for feature in [x for x in self.datasets if x not in ["consumption", "generation"]]:
                if len(self.datasets[feature]["keep"]):
                    csv_path = os.path.join(self.path, 'augmented/' + ftype + '/' + feature + '.csv')
                    df = pd.read_csv(csv_path)
                    timesteps, keys = df.shape
                    assert self.data_info["timesteps"] == 0 or self.data_info["timesteps"] == timesteps, "Timesteps do not match in files (# of rows)"

                    if not self.data_info["features"][feature]:
                        self.data_info["timesteps"] = timesteps
                        self.data_info["features"][feature] = {
                            "start" : self.data_info["keys"],
                            "end" : self.data_info["keys"] + keys,
                            "df" : defaultdict(pd.DataFrame)
                        }

                    if f1 == 0:
                        self.data_info["keys"] += keys
                        self.data_info["labels"].extend([feature + "_" + x for x in df.columns])

                    self.data_info["features"][feature]["df"][ftype] = df

            for etype in ["consumption", "generation"]:
                if len(self.datasets[etype]):
                    for h1, household in enumerate(self.households):
                        csv_path = os.path.join(self.path, 'augmented/' + ftype + '/households/' + household + '_' + etype + '.csv')
                        df = pd.read_csv(csv_path)
                        timesteps, keys = df.shape
                        
                        assert self.data_info["timesteps"] == timesteps, "Timesteps in household data does not match (# of rows)"
                        if not self.data_info["features"][etype]:
                            self.data_info["features"][etype] = {
                                "start" : self.data_info["keys"],
                                "end" : self.data_info["keys"] + keys,
                                "df" : {
                                    "original" : defaultdict(pd.DataFrame),
                                    "adjusted" : defaultdict(pd.DataFrame)
                                }
                            }

                        if f1 == 0 and h1 == 0: # Only increase key number when etype changes (not ftype or household)
                            self.data_info["keys"] += keys
                            self.data_info["labels"].extend([etype + "_" + x for x in df.columns])
                        
                        self.data_info["features"][etype]["df"][ftype][household] = df
        
        self.data = {
            "original" : np.empty((len(self.households), self.data_info["timesteps"], self.data_info["keys"]), dtype=np.float32),
            "adjusted" : np.empty((len(self.households), self.data_info["timesteps"], self.data_info["keys"]), dtype=np.float32)
        }

        for ftype in ["original", "adjusted"]:
            for feature in [x for x in self.data_info["features"] if x not in ["consumption", "generation"]]:
                start = self.data_info["features"][feature]["start"]
                end = self.data_info["features"][feature]["end"]
                self.data[ftype][:, :, start:end] = self.data_info["features"][feature]["df"][ftype] # copies over all households

        for ftype in ["original", "adjusted"]:
            for etype in [x for x in self.data_info["features"] if x in ["consumption", "generation"]]:
                for h, household in enumerate(self.households):
                    start = self.data_info["features"][etype]["start"]
                    end = self.data_info["features"][etype]["end"]
                    self.data[ftype][h, :, start:end] = self.data_info["features"][etype]["df"][ftype][household]

        self.data_info["episodes"] = int(np.ceil(self.data_info["timesteps"] / self.episode_len)) # episodes per household

        # calculate timesteps in final episode (as it may not be a full ep)
        final_ep_steps = self.data_info["timesteps"] - (self.episode_len * int(self.data_info["timesteps"] / self.episode_len))
        if final_ep_steps < self.episode_len:
            logging.info('Final episode has a length of {} rather than {}'.format(final_ep_steps, self.episode_len))
        if final_ep_steps < self.cutoff_episode_length:
            logging.info('Removing final episode (cutoff-length is {} timesteps)'.format(self.cutoff_episode_length))
            self.data_info["episodes"] -= 1
            self.data_info["timesteps"] = self.episode_len * self.data_info["episodes"]
            for ftype in ["original", "adjusted"]:
                # remove final timesteps (part-episode)
                self.data[ftype] = self.data[ftype][:, :self.data_info["timesteps"]]

        return True

    def create_directory_structure(self):
        # This is the intended structure of the "data/" directory
        dirs = {
            # this houses the processed data (adjusted refers to mean/std adjusted)
            'augmented' : {
                'adjusted' : {
                    'households' : {}
                },
                'original' : {
                    'households' : {}
                }
            },
            # this houses the actual pre-processed data
            'original' : {
                'households' : {}
            }
        }

        # converts a nested directory dictionary to a list
        # only saves the deepest directories (others are implied by path structure)
        def dir_dict_to_list(dirs, str_root=''):
            ldirs = []
            for root in dirs:
                cur_str_root = '' if not str_root else str_root + "/"
                cur_str_root = cur_str_root + root
                if dirs[root]:
                    # recurse through dictionary
                    ldirs.extend(dir_dict_to_list(dirs[root], cur_str_root))
                else:
                    # we hit the end node, add to our overall list!
                    ldirs.append(cur_str_root)

            return ldirs

        # creates directionaries given in a list
        def create_directories(dirs):
            for _dir in dirs:
                path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/" + _dir)
                os.makedirs(path, exist_ok=True)

        dirs = dir_dict_to_list(dirs)
        create_directories(dirs)

    def check_augmented_data(self):
        
        # TODO: This is not a very involved check! 

        # First create directory structure if it doesn't already exist
        self.create_directory_structure()

        # The following checks that all augmented data exists, returns False if not
        # Only checks files which are used (i.e. where `keep` is not empty)

        csv_files_loaded = 0

        for ftype in ["original", "adjusted"]:
            
            for household_id in self.households:
                for etype in ["consumption", "generation"]:
                    
                    if len(self.datasets[etype]["keep"]):
                        csv_path = os.path.join(self.path, 'augmented/' + ftype + '/households/' + household_id + '_' + etype + '.csv')
                        if not os.path.isfile(csv_path):
                            raise FileNotFoundError('Augmented household data file does not exist: {}'.format(csv_path))
                        csv_files_loaded += 1

            for key in self.datasets:
                if key not in ["consumption", "generation"] and len(self.datasets[key]["keep"]):
                    csv_path = os.path.join(self.path, 'augmented/' + ftype + '/' + key + '.csv')
                    if not os.path.isfile(csv_path):
                        raise FileNotFoundError('Augmented data file does not exist: {}'.format(csv_path))
                    csv_files_loaded += 1

        if csv_files_loaded == 0:
            raise Exception("Augmented data is empty! No data has been kept")

    def augment_data(self):
        
        augmented_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data/augmented")
        logging.info('Removing augmented data directory')
        shutil.rmtree(augmented_path)
        self.create_directory_structure()

        self.data = {
            "original" : defaultdict(pd.DataFrame),
            "adjusted" : defaultdict(pd.DataFrame)
        }

        self.energy_data = {
            "consumption" : {
                "original" : defaultdict(pd.DataFrame),
                "adjusted" : defaultdict(pd.DataFrame)
            },
            "generation" : {
                "original" : defaultdict(pd.DataFrame),
                "adjusted" : defaultdict(pd.DataFrame)
            }
        }

        # standardise demand/price/weather data
        for dataset in self.datasets:
             # ignore household data here (handled separately, below)
             # Also only load in dataset if we want to use it (i.e. keep is not empty)
            if dataset not in ["consumption", "generation"] and len(self.datasets[dataset]["keep"]):
                csv_path = os.path.join(self.path, 'original/' + dataset + '.csv')
                logging.info('Loading in data %s', csv_path)
                self.data["original"][dataset] = pd.read_csv(csv_path,
                                                             usecols=self.datasets[dataset]["keep"])

                # convert from W to kWh
                if dataset is "demand":
                    for key in [x for x in self.data["original"][dataset] if x not in ["date"]]:
                        # 60000 = 1000 (W -> kW) * 60 (kWh for 1 minute)
                        self.data["original"][dataset][key] /= 60000
                        # CSV files were incorrectly multiplied by 60, so reverse this
                        self.data["original"][dataset][key] /= 60

                self.data["adjusted"][dataset] = self.data["original"][dataset].copy()

                # mean/std adjust all floats
                for key in self.data["adjusted"][dataset].select_dtypes(include=['float']):
                    data = self.data["adjusted"][dataset][key]
                    mean = np.mean(data)
                    std = np.std(data)
                    adjusted = 0.0 if mean == 0.0 and std == 0.0 else (self.data["adjusted"][dataset][key] - mean) / std
                    self.data["adjusted"][dataset][key] = adjusted
                    
                for ftype in ["original", "adjusted"]:
                    # remove date cols
                    if "date" in self.data[ftype][dataset]:
                        self.data[ftype][dataset].drop(columns=["date"], inplace=True)
                    csv_path = os.path.join(self.path, 'augmented/' + ftype + '/' + dataset + '.csv')
                    logging.info('Saving data %s', csv_path)
                    self.data[ftype][dataset].to_csv(csv_path, index=False)

        for household_id in self.households:
            csv_path = os.path.join(self.path, 'original/households/' + household_id + '.csv')
            logging.info('Loading in household data %s', csv_path)

            # read all columns from the household data (split later)
            household_data = pd.read_csv(csv_path)

            # null statistics
            nans = household_data.isnull().sum()
            total_nans = nans.sum()
            assert nans['date'] == 0, "Missing dates in household {} file".format(household_id)

            # format the date column (originally just an 'object')
            household_data["date"] = pd.to_datetime(household_data['date'], format='%Y-%m-%d %H:%M:%S')
            # TODO: Allow this column to be used for something (currently it's just removed)

            household_data["weekday"] = (household_data["date"].dt.weekday >= 5).astype(int)

            # remove date column
            household_data.drop(columns=["date"], inplace=True)

            # Fill in NaN values and report this
            # TODO: Make this more involved (e.g. average of neighbouring vals) - bfill, ffill may be useful
            if total_nans > 0:
                logging.info("NaNs in household %s (%d total):\n\n%s", household_id, total_nans, nans)
                household_data = household_data.fillna(household_data.mean())

            # Add in missing columns (missing appliances have a consumption/generation of 0)
            all_appliances = self.datasets["consumption"]["list"] + self.datasets["generation"]["list"]
            for key in all_appliances:
                if key not in household_data:
                    household_data[key] = 0.
                    
            # Convert W to kWh (divide by 60000 as duration is over one minute <- 60*1000)
            household_data /= 60000.0
            # CSV files were incorrectly multiplied by 60, so reverse this
            household_data /= 60.0
                
            # Split appliances into consumption and generation
            for etype in ["consumption", "generation"]:
                self.energy_data[etype]["original"][household_id] = household_data[self.datasets[etype]["list"]].copy()
            
            # Make generation positive
            self.energy_data["generation"]["original"][household_id] *= -1

            # Add a total column to generation
            self.energy_data["generation"]["original"][household_id]['total'] = self.energy_data["generation"]["original"][household_id].select_dtypes(['float']).sum(axis=1)

            # Rename main column to total for consumption
            self.energy_data["consumption"]["original"][household_id].rename(columns={"main":"total"}, inplace=True)

            # Add an `other` field to consumption (difference between total and sum of NILM)
            consumption_appliances = [x for x in self.datasets["consumption"]["list"] if x not in ["main"]]
            self.energy_data["consumption"]["original"][household_id]['other'] = \
                self.energy_data["consumption"]["original"][household_id]['total'] \
                - self.energy_data["consumption"]["original"][household_id][consumption_appliances].sum(axis=1)

            # Remove unwanted data
            for etype in ["consumption", "generation"]:
                self.energy_data[etype]["original"][household_id] = self.energy_data[etype]["original"][household_id][self.datasets[etype]["keep"]]

            # normalise data
            for etype in ["consumption", "generation"]:
                for household_id in self.energy_data[etype]["original"]:
                    for key in self.energy_data[etype]["original"][household_id].select_dtypes(['float', 'int']):
                        data = self.energy_data[etype]["original"][household_id][key]
                        mean = np.mean(data)
                        std = np.std(data)
                        # weird subtraction necessary to avoid nan (0.0 doesn't cut it)
                        adjusted = (self.energy_data[etype]["original"][household_id][key] - self.energy_data[etype]["original"][household_id][key]) if mean == 0.0 and std == 0.0 else (self.energy_data[etype]["original"][household_id][key] - mean) / std
                        self.energy_data[etype]["adjusted"][household_id][key] = adjusted

                # save to csv for quick loading in future
                for ftype in ["original", "adjusted"]:
                    csv_path = os.path.join(self.path, 'augmented/' + ftype + '/households/' + household_id + '_' + etype + '.csv')
                    logging.info('Saving household data %s', csv_path)
                    self.energy_data[etype][ftype][household_id].to_csv(csv_path, index=False)

    def load_episode(self, rand=True, episode=0, household=0, train=True):

        # TODO: Why have we forced episodes to start strictly at a specific position? Could just load a random batch of x consecutive timesteps?
        # TODO: add in some date-features (day number, e.g.)

        # if train is True then we limit episodes between [0, cutoff]
        # otherwise [cutoff+1, end]. cutoff given by self.train_test

        limits = [0, self.cutoff-1] if train else [self.cutoff, self.data_info["episodes"]-1]

        if rand:
            household = random.randint(0, len(self.households)-1)
            episode = random.randint(*limits)

        start = self.episode_len * episode
        end = np.minimum(self.episode_len * (episode + 1), self.data_info["timesteps"])

        self.current_episode = {
            "household" : household,
            "episode" : episode,
            "timestep" : 0,
            "start" : start,
            "end" : end
        }

    def _access_episode_data(self, key, ftype):
        if key in ["consumption", "generation"]:
            return self.energy_data[key][ftype][self.current_episode["household"]][self.current_episode["episode"]]
        else:
            return self.data[ftype][key][self.current_episode["episode"]]

    def next(self):

        start_timestep = self.current_episode["timestep"] + self.current_episode["start"]
        end_timestep = start_timestep + self.foresight + 1

        if end_timestep > self.current_episode["end"]: # TODO: verify this is correct
            raise StopIteration
            
        adjusted_data = self.data["adjusted"][self.current_episode["household"], start_timestep:end_timestep]
        adjusted_data = adjusted_data.reshape(-1) # collapse into one list (# of features = # keys * (foresight+1))
        
        original_data = self.data["original"][self.current_episode["household"], start_timestep:end_timestep]
        original_data = original_data.reshape(-1) # collapse into one list (# of features = # keys * (foresight+1))

        self.current_episode["timestep"] += 1

        return original_data, adjusted_data

    def get_data(self, household=0):

        original_data = self.data["original"][household]

        return {
            "household" : {
                "data" : original_data,
                "labels" : self.data_info['labels']
            },
            "price" : self.data_info["features"]["price"]["df"]["original"]
        }