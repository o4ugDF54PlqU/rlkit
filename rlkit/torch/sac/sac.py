from collections import OrderedDict, namedtuple
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim
from rlkit.core.loss import LossFunction, LossStatistics
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from rlkit.core.logging import add_prefix
import gtimer as gt
import os
import random

SACLosses = namedtuple(
    'SACLosses',
    'policy_loss qf1_loss qf2_loss alpha_loss state_estimator_loss',
)

# Active Soft Actor Critic
class ASACTrainer(TorchTrainer, LossFunction):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            state_estimator,

            discount=0.99,
            reward_scale=1.0,

            cost=1e-4,  # Measurement Cost
            state_estimator_lr=1e-3,
            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,
            replay=False,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
    ):
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.state_estimator = state_estimator
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period
        self.cost = cost
        self.replay = replay

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy is None:
                # Use heuristic value from SAC paper
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()
            else:
                self.target_entropy = target_entropy
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.state_estimator_criterion = nn.MSELoss()
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.state_estimator_optimizer = optimizer_class(
            self.state_estimator.parameters(),
            lr=state_estimator_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.eval_statistics = OrderedDict()

        if replay and os.path.exists("observations.txt"):
            # Read in buffer for training ASAC with "expert" data
            observations = torch.Tensor(np.loadtxt("observations.txt")).cuda()
            actions = torch.Tensor(np.loadtxt("actions.txt")).cuda()
            next_observations = torch.Tensor(np.loadtxt("next_observations.txt")).cuda()
            self.state_estimator = self.state_estimator.cuda()
            # If we decide to add a loop instead of a single gradient descent, make here
            state_estimator_pred = self.state_estimator(observations, actions)
            state_estimator_loss = self.state_estimator_criterion(state_estimator_pred, next_observations)
            self.state_estimator_optimizer.zero_grad()
            state_estimator_loss.backward()
            self.state_estimator_optimizer.step()

    def train_from_torch(self, batch):
        # This is the entry point for training for AsSAC
        gt.blank_stamp()
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        """
        Update networks
        """
        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            losses.alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        losses.policy_loss.backward()
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        losses.qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        losses.qf2_loss.backward()
        self.qf2_optimizer.step()

        self.state_estimator_optimizer.zero_grad()
        losses.state_estimator_loss.backward()
        self.state_estimator_optimizer.step()

        self._n_train_steps_total += 1

        self.try_update_target_networks()
        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        gt.stamp('sac training', unique=False)

    def try_update_target_networks(self):
        if self._n_train_steps_total % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(
            self.qf1, self.target_qf1, self.soft_target_tau
        )
        ptu.soft_update_from_to(
            self.qf2, self.target_qf2, self.soft_target_tau
        )

    def compute_loss(
        self,
        batch,
        skip_statistics=False,
    ) -> Tuple[SACLosses, LossStatistics]:
        rewards = batch['rewards'] # torch.Size([256, 1])
        terminals = batch['terminals']
        obs = batch['observations'] # torch.Size([256, 17])
        actions = batch['actions'] # torch.Size([256, 7])
        actions_without_measure = actions[:,:-1] #  [256, 6]
        next_obs = batch['next_observations']

        next_obs_only_measure = torch.Tensor([]).cuda()
        obs_only_measure = torch.Tensor([]).cuda()
        actions_without_measure_only_measure = torch.Tensor([]).cuda()
        measured = False

        # Calculate costs based on measure/non-measure
        costs = torch.zeros(rewards.size()).cuda()
        for i in range(len(rewards)):
            if actions[i][-1] > 0.0: # Range is (-1, 1); (0, 1) is measure
                measured = True
                costs[i] = self.cost
                
                next_obs_only_measure = torch.cat((next_obs_only_measure, next_obs[i].unsqueeze(0)))
                obs_only_measure = torch.cat((obs_only_measure, obs[i].unsqueeze(0)))
                actions_without_measure_only_measure = torch.cat((actions_without_measure_only_measure,
                                                                actions_without_measure[i].unsqueeze(0)))

        """
        Policy and Alpha Loss
        """
        dist = self.policy(obs) # Gets distribution for stochastic action
        new_obs_actions, log_pi = dist.rsample_and_logprob() # Chooses action from distribution
        log_pi = log_pi.unsqueeze(-1)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        ) # Finds min-Q value from both Q tables for this action
        policy_loss = (alpha*log_pi - q_new_actions).mean()

        """
        State Estimator Loss
        """
        # obs.shape = torch.Size([256, 17]), type = torch.Tensor
        # actions_without_measure.shape = torch.Size([256, 6]), type = torch.Tensor
        # state_estimator_pred = self.state_estimator(obs, actions_without_measure)
        # state_estimator_loss = self.state_estimator_criterion(state_estimator_pred, next_obs)

        # print("obs_only_measure size", obs_only_measure.size())
        # print("action_too_long size", actions_without_measure_only_measure.size())
        # print("obs_only_measure", obs_only_measure)
        # print("action_too_long", actions_without_measure_only_measure)
        if not measured:
            rand = random.randrange(len(rewards))
            next_obs_only_measure = torch.cat((next_obs_only_measure, next_obs[rand].unsqueeze(0)))
            obs_only_measure = torch.cat((obs_only_measure, obs[rand].unsqueeze(0)))
            actions_without_measure_only_measure = torch.cat((actions_without_measure_only_measure,
                                                            actions_without_measure[rand].unsqueeze(0)))
        state_estimator_pred = self.state_estimator(obs_only_measure, actions_without_measure_only_measure)
        state_estimator_loss = self.state_estimator_criterion(state_estimator_pred, next_obs_only_measure)

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        next_dist = self.policy(next_obs)
        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(-1)
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * (rewards - costs) + (1. - terminals) * self.discount * target_q_values # Update with Cost
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics['State Estimator Loss'] = np.mean(ptu.get_numpy(state_estimator_loss))
            eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
            eval_statistics.update(policy_statistics)
            if self.use_automatic_entropy_tuning:
                eval_statistics['Alpha'] = alpha.item()
                eval_statistics['Alpha Loss'] = alpha_loss.item()

        loss = SACLosses(
            policy_loss=policy_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            alpha_loss=alpha_loss,
            state_estimator_loss=state_estimator_loss,
        )

        return loss, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
            self.state_estimator,
        ]

    @property
    def optimizers(self):
        return [
            self.alpha_optimizer,
            self.qf1_optimizer,
            self.qf2_optimizer,
            self.policy_optimizer,
            self.state_estimator_optimizer,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )