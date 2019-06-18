from sim.agents.interface import IRLUnit

import copy
import glob
import os
import time
import types

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_ppo.arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from torch_ppo.envs import make_env
from torch_ppo.model import Policy
from torch_ppo.storage import RolloutStorage
from torch_ppo.utils import update_current_obs
from torch_ppo.visualize import visdom_plot

from collections import defaultdict

import torch_ppo.algo as algo

"""
PPO agent (uses pytorchrl implementation)
"""

class PPO(IRLUnit):

    name = "PPO"

    def __init__(self, results):
        self.results = results
        self._act = None
        self.sess = None

    def train(self,
              env,
              episode_length=None,
              recurrent_policy=False,
              num_processes=8,
              num_steps=128,
              seed=1,
              port=8097,
              vis=False,
              gamma=1.00,
              num_stack=1,
              clip_param=0.1,
              ppo_epoch=4,
              num_mini_batch=4,
              value_loss_coef=1,
              entropy_coef=0.01,
              lr=7e-5,
              epsilon=1e-5,
              max_grad_norm=0.5,
              use_gae=True,
              tau=0.95,
              save_interval=100,
              log_interval=1,
              vis_interval=1,
              env_name="Battery",
              early_stop=5):

        # Ensure required params are passed
        if episode_length is None:
            raise ValueError("Required parameters not passed")

        # Set default values for various agent parameters
        num_frames = 1.5e6
        
        min_num_updates = int(num_frames) // num_steps // num_processes
        
        cuda = torch.cuda.is_available()
        torch.manual_seed(seed)
        if cuda:
            torch.cuda.manual_seed(seed)

        torch.set_num_threads(1)

        if vis:
            from visdom import Visdom
            viz = Visdom(port=port)
            win = None

        self.results.create_dir("log")
        log_dir = self.results.get_path("log/")
        try:
            os.makedirs(log_dir)
        except OSError:
            files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
            for f in files:
                os.remove(f)

        envs_original = [make_env(env, seed, i, log_dir, False)
                for i in range(num_processes)]

        env_name = self.results.current_dir

        if num_processes > 1:
            envs = SubprocVecEnv(envs_original)
        else:
            envs = DummyVecEnv(envs_original)

        if len(envs.observation_space.shape) == 1:
            envs = VecNormalize(envs, gamma=gamma)

        obs_shape = envs.observation_space.shape
        obs_shape = (obs_shape[0] * num_stack, *obs_shape[1:])
        
        actor_critic = Policy(obs_shape, envs.action_space, recurrent_policy)
        
        if envs.action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = envs.action_space.shape[0]

        if cuda:
            actor_critic.cuda()

        agent = algo.PPO(actor_critic, clip_param, ppo_epoch, num_mini_batch,
                         value_loss_coef, entropy_coef, lr=lr,
                               eps=epsilon,
                               max_grad_norm=max_grad_norm)

        rollouts = RolloutStorage(num_steps, num_processes, obs_shape, envs.action_space, actor_critic.state_size)
        current_obs = torch.zeros(num_processes, *obs_shape)
        
        obs = envs.reset()
        update_current_obs(obs, current_obs, obs_shape, num_stack)

        rollouts.observations[0].copy_(current_obs)

        # These variables are used to compute average rewards for all processes.
        episode_rewards = torch.zeros([num_processes, 1])
        final_rewards = torch.zeros([num_processes, 1])

        if cuda:
            current_obs = current_obs.cuda()
            rollouts.cuda()

        save_path = self.results.get_path("model.pt")
        
        start = time.time()

        j = 0

        while True:
            for step in range(num_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, states = actor_critic.act(
                            rollouts.observations[step],
                            rollouts.states[step],
                            rollouts.masks[step])
                cpu_actions = action.squeeze(1).cpu().numpy()

                # Obser reward and next obs
                # Everything returned here is a list over all envs (over CPUs)
                obs, reward, done, info = envs.step(cpu_actions)

                reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
                
                episode_rewards += reward

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks

                if cuda:
                    masks = masks.cuda()

                if current_obs.dim() == 4:
                    current_obs *= masks.unsqueeze(2).unsqueeze(2)
                else:
                    current_obs *= masks

                update_current_obs(obs, current_obs, obs_shape, num_stack)
                rollouts.insert(current_obs, states, action, action_log_prob, value, reward, masks)

            with torch.no_grad():
                next_value = actor_critic.get_value(rollouts.observations[-1],
                                                    rollouts.states[-1],
                                                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, use_gae, gamma, tau)

            value_loss, action_loss, dist_entropy = agent.update(rollouts)

            rollouts.after_update()
            
            if j % save_interval == 0:

                # A really ugly way to save a model to CPU
                save_model = actor_critic
                if cuda:
                    save_model = copy.deepcopy(actor_critic).cpu()

                save_model = [save_model,
                                hasattr(envs, 'ob_rms') and envs.ob_rms or None]

                torch.save(save_model, save_path)
                print("MODEL SAVED")

            if j % log_interval == 0:
                end = time.time()
                total_num_steps = (j + 1) * num_processes * num_steps
                print("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                    format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        final_rewards.mean(),
                        final_rewards.median(),
                        final_rewards.min(),
                        final_rewards.max(), dist_entropy,
                        value_loss, action_loss))

            if vis and j % vis_interval == 0:
                try:
                    # Sometimes monitor doesn't properly flush the outputs
                    win = visdom_plot(viz, win, log_dir, env_name,
                                    algo, num_frames)
                except IOError:
                    pass

            if j > min_num_updates:
                break
                if np.all(averaged['episode_rewards'][-early_stop] > averaged['episode_rewards'][-(early_stop-1):]):
                    break

            j += 1

        y = len(info[0]['episodic']['episode_rewards'])
        averaged = {}

        for episode_data in info:
            for key in episode_data['episodic']:
                if isinstance(episode_data['episodic'][key], list):
                    if key not in averaged:
                        averaged[key] = np.zeros(y)
                    averaged[key] += np.asarray(episode_data['episodic'][key]) / num_processes
                else: # defaultdict
                    if key not in averaged:
                        averaged[key] = defaultdict(lambda: np.zeros(y))
                    for subkey in episode_data['episodic'][key]:
                        averaged[key][subkey] += np.asarray(episode_data['episodic'][key][subkey]) / num_processes

        return averaged

    def play(self,
             env=None,
             data=None,
             timesteps=None,
             indices=None,
             household=None):
        
        seed = 1
        num_stack = 1

        actor_critic, ob_rms = torch.load(self.results.get_path("model.pt"))

        obs_shape = env.observation_space.shape
        obs_shape = (obs_shape[0] * num_stack, *obs_shape[1:])
        current_obs = torch.zeros(1, *obs_shape)
        states = torch.zeros(1, actor_critic.state_size)
        masks = torch.zeros(1, 1)

        requested = np.zeros((timesteps,))
        limits = np.zeros((timesteps,))
        cost = np.zeros((timesteps,))
        prices = np.zeros((timesteps,))
        charges = np.zeros((timesteps,))

        t = 0
        for episode in range(env.data.cutoff, env.data.data_info["episodes"]):

            obs = env.reset(rand=False, episode=episode, household=household, carry_charge=True, train=False)
            update_current_obs(obs, current_obs, obs_shape, num_stack)

            done = False

            while not done:
                with torch.no_grad():
                    value, action, _, states = actor_critic.act(current_obs,
                                                                states,
                                                                masks,
                                                                deterministic=True)
                cpu_actions = action.squeeze(1).cpu().numpy()
                obs, reward, done, info = env.step(cpu_actions)

                masks.fill_(0.0 if done else 1.0)

                if current_obs.dim() == 4:
                    current_obs *= masks.unsqueeze(2).unsqueeze(2)
                else:
                    current_obs *= masks
                update_current_obs(obs, current_obs, obs_shape, num_stack)

                cost[t] = info['cost']
                requested[t] = info['requested']
                limits[t] = info['demand']
                prices[t] = info['price']
                charges[t] = info['charge']
                t += 1
            
            print("Week {} finished testing ({})".format(episode, PPO.name))

        cost = np.sum(cost.reshape((-1, 60*24)).mean(axis=0))

        return cost, requested, limits, prices, charges

    def reset_session(self):
        pass
