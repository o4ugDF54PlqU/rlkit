from random import randint
from gym.envs.mujoco import HalfCheetahEnv

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import ActiveMdpPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.asac import ASACTrainer
from rlkit.torch.networks import ConcatMlp, ConcatEnsembleMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import os
# os.environ['CUDA_VISIBLE_DEVICES']=f"{randint(0,7)}"

def experiment(variant):
    expl_env = NormalizedBoxEnv(HalfCheetahEnv())
    eval_env = NormalizedBoxEnv(HalfCheetahEnv())
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size
    action_dim_with_measure = action_dim + 1
    cost = 1e-3
    replay = "none"

    # Environment and Algorithm Specifications:
    # obs_dim = 17
    # action_dim = 6
    # Critic: (s, a, m) -> value for how good that action / measure/non-measure is in that state
    # Actor: (s) -> (a, m)

    M = variant['layer_size']
    # State Estimator
    state_estimator = ConcatEnsembleMlp(
        hidden_sizes=[M, M],
        output_size=obs_dim,
        input_size=obs_dim + action_dim,
        ensemble_count=3,
        state_estimator_lr=1e-4,
        device=ptu.device
    )
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim_with_measure,
        output_size=1,
        hidden_sizes=[M, M],
    )
    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim_with_measure,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim_with_measure,
        output_size=1,
        hidden_sizes=[M, M],
    )
    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim_with_measure,
        output_size=1,
        hidden_sizes=[M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim_with_measure,
        hidden_sizes=[M, M],
    )

    eval_policy = MakeDeterministic(policy)
    eval_path_collector = ActiveMdpPathCollector(
        eval_env,
        eval_policy,
        state_estimator,
        cost,
    )
    expl_path_collector = ActiveMdpPathCollector(
        expl_env,
        policy,
        state_estimator,
        cost,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    trainer = ASACTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        state_estimator=state_estimator,
        cost=cost,
        replay=replay,
        device=ptu.device,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="ASAC",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(1E6),
        algorithm_kwargs=dict(
            num_epochs=1500,
            num_eval_steps_per_epoch=2500,
            num_trains_per_train_loop=500,
            num_expl_steps_per_train_loop=500,
            min_num_steps_before_training=500,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            target_update_period=1,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,
        ),
    )
    setup_logger('Imback', variant=variant)
    ptu.set_gpu_mode(False)  # optionally set the GPU (default=False)
    experiment(variant)
