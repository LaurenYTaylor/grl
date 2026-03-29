import os
import time
from contextlib import contextmanager

import flax
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils import dlpack as torch_dlpack
import tqdm
from absl import app, flags, logging
from flax.training import checkpoints
from ml_collections import config_flags
from vmas import make_env

from experiments.configs.ensemble_config import add_redq_config
from experiments.configs.multiagent_policy_config import get_config as _get_policy_config
from wsrl.agents import agents
from wsrl.common.wandb import WandBLogger
from wsrl.data.replay_buffer import TorchGPUReplayBuffer, TorchGPUReplayBufferMC
from wsrl.utils.timer_utils import Timer

flax.config.update("flax_use_orbax_checkpointing", True)

FLAGS = flags.FLAGS

ENV_KWARGS = {
    "balance": {"n_agents": 5, "random_package_pos_on_line": False, "max_steps": 50},
    "discovery": {"n_agents": 4, "use_agent_lidar": False, "max_steps": 50},
    "dispersion": {"n_agents": 20, "share_reward": True, "penalise_by_time": False, "max_steps": 50},
    "dropout": {"n_agents": 5, "energy_coeff": 0.02, "max_steps": 50},
    "flocking": {"n_agents": 5, "max_steps": 50},
    "give_way": {"mirror_passage": False, "max_steps": 50},
    "navigation": {"n_agents": 3, "max_steps": 50},
    "passage": {"n_passages": 1, "shared_reward": True, "max_steps": 50},
    "reverse_transport": {"n_agents": 5, "package_width": 0.6, "package_length": 0.6, "package_mass": 50, "max_steps": 50},
    "transport": {"n_agents": 6, "n_packages": 1, "package_width": 0.15, "package_length": 0.15, "package_mass": 50, "max_steps": 50},
    "wheel": {"n_agents": 2, "line_length": 2, "desired_velocity": 0.1, "max_steps": 50},
}

# env
flags.DEFINE_string("env", "transport", "Environment/scenario to use")
flags.DEFINE_float("reward_scale", 1.0, "Reward scale.")
flags.DEFINE_float("reward_bias", -1.0, "Reward bias.")
flags.DEFINE_float(
    "clip_action",
    0.99999,
    "Clip actions to be between [-n, n]. This is needed for tanh policies.",
)
flags.DEFINE_integer("num_envs", 10, "Number of vectorized environments")
flags.DEFINE_integer("num_agents", 4, "Number of agents in the environment")
flags.DEFINE_bool("continuous", True, "Whether actions are continuous")
flags.DEFINE_integer("max_env_steps", 200, "Maximum environment horizon")
flags.DEFINE_string("device", "cuda", "Simulation device")

# training
flags.DEFINE_integer("num_offline_steps", 0, "Initial no-update collection steps")
flags.DEFINE_integer("num_online_steps", 500_000, "Number of online training steps")
flags.DEFINE_float(
    "offline_data_ratio",
    0.0,
    "Unused for multiagent VMAS; kept for config compatibility.",
)
flags.DEFINE_string(
    "online_sampling_method",
    "append",
    "Unused for multiagent VMAS; kept for config compatibility.",
)
flags.DEFINE_bool(
    "online_use_cql_loss",
    True,
    "When agent is CQL/CalQL, whether to use CQL loss online.",
)
flags.DEFINE_integer("warmup_steps", 0, "Minimum transitions before updates")

# agent
flags.DEFINE_string("agent", "sac", "What RL agent to use")
flags.DEFINE_integer("utd", 1, "Update-to-data ratio of the critic")
flags.DEFINE_integer("batch_size", 256, "Batch size for training")
flags.DEFINE_integer("replay_buffer_capacity", int(2e6), "Replay buffer capacity")
flags.DEFINE_bool("use_redq", False, "Use an ensemble of Q-functions")

# experiment house keeping
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string(
    "save_dir",
    os.path.expanduser(os.getcwd() + "/exp_logging"),
    "Directory to save the logs and checkpoints",
)
flags.DEFINE_string("resume_path", "", "Directory to resume checkpoints from")
flags.DEFINE_string("guide_policy_path", "", "Path to restore guide agent from")
flags.DEFINE_integer("n_curriculum_stages", 10, "N curriculum stages")
flags.DEFINE_integer("log_interval", 5_000, "Log every n steps")
flags.DEFINE_integer("eval_interval", 20_000, "Evaluate every n steps")
flags.DEFINE_integer("save_interval", 100_000, "Save every n steps")
flags.DEFINE_integer("n_eval_trajs", 20, "Number of trajectories to evaluate")
flags.DEFINE_bool("deterministic_eval", True, "Whether to use deterministic evaluation")
flags.DEFINE_bool("profile_code", False, "Profile rollout/replay/update hot paths")
flags.DEFINE_integer(
    "profile_log_interval", 1000, "Log profiling statistics every n steps"
)

# wandb
flags.DEFINE_string("exp_name", "", "Experiment name for wandb logging")
flags.DEFINE_string("project", None, "Wandb project folder")
flags.DEFINE_string("group", None, "Wandb group of the experiment")
flags.DEFINE_bool("debug", False, "If true, no logging to wandb")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)



class _CodeProfiler:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._stats = {}

    @contextmanager
    def measure(self, name: str):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            stat = self._stats.setdefault(
                name,
                {"total_s": 0.0, "count": 0, "max_s": 0.0},
            )
            stat["total_s"] += elapsed
            stat["count"] += 1
            stat["max_s"] = max(stat["max_s"], elapsed)

    def summary(self, reset: bool = False):
        result = {}
        for name, stat in self._stats.items():
            if stat["count"] == 0:
                continue
            result[name] = {
                "avg_ms": 1000.0 * stat["total_s"] / stat["count"],
                "total_s": stat["total_s"],
                "count": stat["count"],
            }
        if reset:
            self._stats = {}
        return result


def _sync_torch_device(device=None):
    if torch.cuda.is_available():
        if device is None:
            torch.cuda.synchronize()
        else:
            torch.cuda.synchronize(device)


def _block_until_ready_tree(data):
    if isinstance(data, dict):
        return {k: _block_until_ready_tree(v) for k, v in data.items()}
    if hasattr(data, "block_until_ready"):
        return data.block_until_ready()
    return data


def _torch_to_jax_tree(data):
    if isinstance(data, dict):
        return {k: _torch_to_jax_tree(v) for k, v in data.items()}
    if torch.is_tensor(data):
        return jax.dlpack.from_dlpack(data.contiguous())
    if isinstance(data, (np.ndarray, jnp.ndarray)):
        return jnp.asarray(data)
    return jnp.asarray(data)


def _jax_to_torch_tree(data):
    if isinstance(data, dict):
        return {k: _jax_to_torch_tree(v) for k, v in data.items()}
    if torch.is_tensor(data):
        return data
    return torch_dlpack.from_dlpack(data)


def _make_example_jax_batch(space):
    return jnp.asarray(np.expand_dims(space.sample(), axis=0))


def _tree_leading_dim(tree):
    if torch.is_tensor(tree):
        return tree.shape[0]
    if isinstance(tree, dict):
        first_key = next(iter(tree))
        return _tree_leading_dim(tree[first_key])
    raise TypeError(f"Unsupported tree type: {type(tree)}")


def _concat_torch_trees(trees):
    first = trees[0]
    if torch.is_tensor(first):
        return torch.cat(trees, dim=0)
    if isinstance(first, dict):
        return {
            key: _concat_torch_trees([tree[key] for tree in trees])
            for key in first.keys()
        }
    raise TypeError(f"Unsupported tree type: {type(first)}")


def _split_torch_tree(tree, chunk_size):
    if torch.is_tensor(tree):
        return list(torch.split(tree, chunk_size, dim=0))
    if isinstance(tree, dict):
        split_values = {
            key: _split_torch_tree(value, chunk_size) for key, value in tree.items()
        }
        n_chunks = len(next(iter(split_values.values())))
        return [
            {key: split_values[key][i] for key in split_values.keys()}
            for i in range(n_chunks)
        ]
    raise TypeError(f"Unsupported tree type: {type(tree)}")


def _resolve_agent_policy_mapping(agent_names, policy_config):
    policy_sharing = "shared"
    default_policy = "shared_policy"
    mapping_entries = []
    if policy_config is not None:
        policy_sharing = policy_config.get("policy_sharing", policy_sharing)
        default_policy = policy_config.get("default_policy", default_policy)
        mapping_entries = policy_config.get("agent_policy_mapping", [])

    if policy_sharing == "independent":
        agent_to_policy = {
            agent_name: f"policy_{idx}"
            for idx, agent_name in enumerate(agent_names)
        }
        policy_to_agents = {
            policy_name: [agent_name]
            for agent_name, policy_name in agent_to_policy.items()
        }
        return agent_to_policy, policy_to_agents
    if policy_sharing != "shared":
        raise ValueError(
            f"Unknown policy_sharing mode '{policy_sharing}'. Expected 'shared' or 'independent'."
        )

    agent_to_policy = {agent_name: default_policy for agent_name in agent_names}
    for item in mapping_entries:
        if not isinstance(item, dict) or len(item) != 1:
            raise ValueError(
                "Each item in agent_policy_mapping must be a single-entry dictionary."
            )
        agent_name, policy_name = next(iter(item.items()))
        if agent_name not in agent_to_policy:
            raise ValueError(f"Unknown agent in policy mapping: {agent_name}")
        agent_to_policy[agent_name] = policy_name

    policy_to_agents = {}
    for agent_name in agent_names:
        policy_name = agent_to_policy[agent_name]
        policy_to_agents.setdefault(policy_name, []).append(agent_name)
    return agent_to_policy, policy_to_agents


def _sample_env_actions(
    learning_policies, policy_to_agents, observations, env_step, rng, deterministic, profiler=None
):
    env_step = _torch_to_jax_tree(env_step)
    profiler = profiler or _CodeProfiler(False)
    torch_actions = {}
    for policy_name, grouped_agents in policy_to_agents.items():
        grouped_observations = _concat_torch_trees(
            [observations[agent_name] for agent_name in grouped_agents]
        )
        chunk_size = _tree_leading_dim(observations[grouped_agents[0]])
        with profiler.measure("bridge.obs_to_jax"):
            jax_observations = _torch_to_jax_tree(grouped_observations)
            if profiler.enabled:
                _block_until_ready_tree(jax_observations)
        with profiler.measure("policy.sample_actions"):
            if deterministic:
                policy_actions = learning_policies[policy_name].sample_actions(
                    jax_observations,
                    env_step,
                    argmax=True,
                )
            else:
                rng, agent_rng = jax.random.split(rng)
                policy_actions = learning_policies[policy_name].sample_actions(
                    jax_observations,
                    env_step,
                    seed=agent_rng,
                )
            if profiler.enabled:
                policy_actions = _block_until_ready_tree(policy_actions)
        with profiler.measure("bridge.action_to_torch"):
            policy_actions = torch.clamp(
                _jax_to_torch_tree(policy_actions),
                -FLAGS.clip_action,
                FLAGS.clip_action,
            )
            if profiler.enabled:
                _sync_torch_device(policy_actions.device)
        split_actions = _split_torch_tree(policy_actions, chunk_size)
        for agent_name, agent_actions in zip(grouped_agents, split_actions):
            torch_actions[agent_name] = agent_actions
    return torch_actions, rng


def _reset_done_envs(env, observations, done_mask, profiler=None):
    profiler = profiler or _CodeProfiler(False)
    done_indices = torch.nonzero(done_mask, as_tuple=False).flatten().tolist()
    for env_idx in done_indices:
        with profiler.measure("env.reset_at"):
            reset_obs = env.reset_at(int(env_idx))
            if profiler.enabled:
                _sync_torch_device(getattr(env, "device", None))
        for name in observations:
            reset_value = reset_obs[name]
            if (
                getattr(reset_value, "ndim", 0) > 0
                and reset_value.shape[0] == observations[name].shape[0]
            ):
                observations[name][env_idx] = reset_value[env_idx]
            else:
                observations[name][env_idx] = reset_value
    return observations


def _evaluate(eval_env, learning_policies, policy_to_agents, rng, profiler=None):
    profiler = profiler or _CodeProfiler(False)
    with profiler.measure("eval.reset"):
        observations = eval_env.reset()
        if profiler.enabled:
            _sync_torch_device(getattr(eval_env, "device", None))
    all_done = torch.full((eval_env.num_envs,), False, device=eval_env.device)
    episode_returns = torch.zeros((eval_env.num_envs), device=eval_env.device)
    episode_length = torch.zeros(eval_env.num_envs, device=eval_env.device)
    total_return = 0.0
    while (not all_done.all()):
        print(f"Evaluate: {episode_length.min()}/{eval_env.max_steps}")
        actions, rng = _sample_env_actions(
            learning_policies,
            policy_to_agents,
            observations,
            episode_length,
            rng,
            FLAGS.deterministic_eval,
            profiler=profiler,
        )
        #with profiler.measure("eval.step"):
        next_observations, rewards, terminated, truncated, _ = eval_env.step(
            actions
        )
        episode_returns[~all_done] += torch.stack([rewards[name] for name in rewards.keys()], dim=0).mean(dim=0)[~all_done]
        if profiler.enabled:
            _sync_torch_device(getattr(eval_env, "device", None))
        dones = torch.logical_or(terminated, truncated)
        if dones.any():
            all_done += dones
        episode_length += ~dones
        observations = next_observations
    
    total_return = torch.mean(torch.mean(episode_returns, dim=0)).item()
    episode_length = torch.mean(episode_length).item()

    return {
        "average_return": total_return,
        "average_traj_length": episode_length,
    }, rng


def _num_ready_samples(replay_buffer):
    if hasattr(replay_buffer, "_allow_idxs"):
        return len(replay_buffer._allow_idxs)
    return len(replay_buffer)


def main(_):
    assert FLAGS.online_sampling_method in ["mixed", "append"], "incorrect online sampling method"

    if FLAGS.use_redq:
        FLAGS.config.agent_kwargs = add_redq_config(FLAGS.config.agent_kwargs)

    if "guide_policy_path" in FLAGS.config.agent_kwargs:
        FLAGS.config.agent_kwargs["guide_policy_path"] = FLAGS.guide_policy_path

    if "n_curriculum_stages" in FLAGS.config.agent_kwargs:
        FLAGS.config.agent_kwargs["n_curriculum_stages"] = FLAGS.n_curriculum_stages

    if FLAGS.debug:
        wandb_logger = None
        save_dir = os.path.join(FLAGS.save_dir, "debug")
        logging.info("Debug mode enabled: skipping WandB initialization and checkpoint saving.")
    else:
        wandb_config = WandBLogger.get_default_config()
        wandb_config.update(
            {
                "project": FLAGS.project or "wsrl",
                "group": FLAGS.group or "wsrl",
                "exp_descriptor": f"{FLAGS.exp_name}_{FLAGS.env}_{FLAGS.agent}_seed{FLAGS.seed}",
            }
        )

        wandb_logger = WandBLogger(
            wandb_config=wandb_config,
            variant=FLAGS.config.to_dict(),
            random_str_in_identifier=True,
            disable_online_logging=False,
            wandb_output_dir=FLAGS.save_dir,
        )

        save_dir = os.path.join(
            FLAGS.save_dir,
            wandb_logger.config.project,
            f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
        )

    device = torch.device("cuda:1")
    finetune_env = make_env(
        scenario=FLAGS.env,
        num_envs=FLAGS.num_envs,
        device=FLAGS.device,
        continuous_actions=FLAGS.continuous,
        dict_spaces=True,
        terminated_truncated=True,
        seed=FLAGS.seed,
        **ENV_KWARGS.get(FLAGS.env, {}))
    eval_env = make_env(
        scenario=FLAGS.env,
        num_envs=FLAGS.n_eval_trajs,
        device=FLAGS.device,
        continuous_actions=FLAGS.continuous,
        dict_spaces=True,
        terminated_truncated=True,
        seed=FLAGS.seed + 1000,
        **ENV_KWARGS.get(FLAGS.env, {}),
    )
    

    agent_names = list(finetune_env.observation_space.spaces.keys())
    
    FLAGS.clip_action = finetune_env.agents[0].u_range

    replay_buffer_cls = TorchGPUReplayBufferMC if FLAGS.agent == "calql" else TorchGPUReplayBuffer
    policy_config = _get_policy_config(FLAGS.env)
    agent_to_policy, policy_to_agents = _resolve_agent_policy_mapping(
        agent_names, policy_config
    )
    logging.info("Agent to policy mapping: %s", agent_to_policy)

    rng = jax.random.PRNGKey(FLAGS.seed)
    learning_policies = {}
    replay_buffers = {}

    for policy_name, grouped_agents in policy_to_agents.items():
        reference_agent = grouped_agents[0]
        replay_buffers[policy_name] = replay_buffer_cls(
            finetune_env.observation_space.spaces[reference_agent],
            finetune_env.action_space.spaces[reference_agent],
            capacity=FLAGS.replay_buffer_capacity,
            seed=FLAGS.seed,
            discount=FLAGS.config.agent_kwargs.discount if FLAGS.agent == "calql" else None,
            device=FLAGS.device,
        )

        rng, construct_rng = jax.random.split(rng)
        example_obs = _make_example_jax_batch(
            finetune_env.observation_space.spaces[reference_agent]
        )
        example_action = _make_example_jax_batch(
            finetune_env.action_space.spaces[reference_agent]
        )
        learning_policies[policy_name] = agents[FLAGS.agent].create(
            rng=construct_rng,
            observations=example_obs,
            actions=example_action,
            encoder_def=None,
            **FLAGS.config.agent_kwargs,
        )

    if FLAGS.resume_path:
        assert os.path.exists(FLAGS.resume_path), "resume path does not exist"
        for policy_name in policy_to_agents.keys():
            learning_policies[policy_name] = checkpoints.restore_checkpoint(
                FLAGS.resume_path,
                target=learning_policies[policy_name],
                prefix=f"{policy_name}_",
            )

    timer = Timer()
    profiler = _CodeProfiler(FLAGS.profile_code)
    first_policy_name = next(iter(policy_to_agents.keys()))
    step = int(learning_policies[first_policy_name].state.step)
    with profiler.measure("env.reset"):
        observations = finetune_env.reset()
    update_info = {}

    total_steps = FLAGS.num_offline_steps + FLAGS.num_online_steps
    for _ in tqdm.tqdm(range(step, total_steps)):
        if FLAGS.agent in ("cql", "calql") and step == FLAGS.num_offline_steps:
            online_agent_configs = {
                "cql_alpha": FLAGS.config.agent_kwargs.get("online_cql_alpha", None),
                "use_cql_loss": FLAGS.online_use_cql_loss,
            }
            for policy_name in policy_to_agents.keys():
                learning_policies[policy_name].update_config(online_agent_configs)

        timer.tick("total")

        with timer.context("env step"):
            with profiler.measure("rollout.sample_env_actions"):
                env_actions, rng = _sample_env_actions(
                    learning_policies,
                    policy_to_agents,
                    observations,
                    step,
                    rng,
                    deterministic=False,
                    profiler=profiler,
                )
            with profiler.measure("env.step"):
                next_observations, rewards, terminated, truncated, _ = finetune_env.step(
                    env_actions
                )
                if profiler.enabled:
                    _sync_torch_device(getattr(finetune_env, "device", None))

            dones = torch.logical_or(terminated, truncated)
            time_steps = torch.full(
                (FLAGS.num_envs,), step, dtype=torch.int32, device=terminated.device
            )

            for policy_name, grouped_agents in policy_to_agents.items():
                with profiler.measure("replay.insert_batch"):
                    replay_buffers[policy_name].insert_batch(
                        dict(
                            observations=_concat_torch_trees(
                                [observations[name] for name in grouped_agents]
                            ),
                            next_observations=_concat_torch_trees(
                                [next_observations[name] for name in grouped_agents]
                            ),
                            actions=_concat_torch_trees(
                                [env_actions[name] for name in grouped_agents]
                            ),
                            rewards=torch.cat(
                                [rewards[name].to(dtype=torch.float32) for name in grouped_agents],
                                dim=0,
                            ),
                            masks=torch.cat(
                                [torch.logical_not(dones) for _ in grouped_agents], dim=0
                            ),
                            dones=torch.cat(
                                [dones.to(dtype=torch.float32) for _ in grouped_agents], dim=0
                            ),
                            ts=torch.cat([time_steps for _ in grouped_agents], dim=0),
                        )
                    )
                    if profiler.enabled:
                        _sync_torch_device(replay_buffers[policy_name].device)

            if bool(torch.any(dones).item()):
                next_observations = _reset_done_envs(
                    finetune_env, next_observations, dones, profiler=profiler
                )

            observations = next_observations

        with timer.context("update"):
            min_buffer_size = min(
                _num_ready_samples(replay_buffers[policy_name])
                for policy_name in policy_to_agents.keys()
            )
            print("Samples in buffer: ", min_buffer_size)
            updates_enabled = (
                step >= FLAGS.num_offline_steps
                and min_buffer_size >= max(FLAGS.warmup_steps, FLAGS.batch_size)
            )

            if updates_enabled:
                update_info = {}
                for policy_name in policy_to_agents.keys():
                    print("Updating policy:", policy_name)
                    with profiler.measure("replay.sample"):
                        batch = replay_buffers[policy_name].sample(FLAGS.batch_size)
                        if profiler.enabled:
                            batch = _block_until_ready_tree(batch)
                    if FLAGS.utd > 1:
                        with profiler.measure("agent.update_high_utd"):
                            learning_policies[policy_name], agent_info = learning_policies[policy_name].update_high_utd(
                                batch,
                                utd_ratio=FLAGS.utd,
                            )
                            if profiler.enabled:
                                agent_info = _block_until_ready_tree(agent_info)
                    else:
                        with profiler.measure("agent.update"):
                            learning_policies[policy_name], agent_info = learning_policies[policy_name].update(
                                batch,
                            )
                            if profiler.enabled:
                                agent_info = _block_until_ready_tree(agent_info)
                    update_info[policy_name] = agent_info

        step += 1

        eval_steps = (
            FLAGS.num_offline_steps,
            FLAGS.num_offline_steps + 1,
            FLAGS.num_offline_steps + FLAGS.num_online_steps,
        )
        if step % FLAGS.eval_interval == 0 or step in eval_steps:
            logging.info("Evaluating...")
            with timer.context("evaluation"):
                with profiler.measure("evaluation.total"):
                    eval_info, rng = _evaluate(
                        eval_env,
                        learning_policies,
                        policy_to_agents,
                        rng,
                        profiler=profiler,
                    )
                if not FLAGS.debug:
                    wandb_logger.log({"evaluation": eval_info}, step=step)

                for policy_name in policy_to_agents.keys():
                    if hasattr(learning_policies[policy_name], "eval_callback"):
                        learning_policies[policy_name] = learning_policies[policy_name].eval_callback(
                            eval_info["average_return"],
                            np.floor(eval_info["average_traj_length"]),
                        )
                        new_cmprtr = getattr(learning_policies[policy_name], "cmprtr", 0.0)
                        if not FLAGS.debug:
                            wandb_logger.log(
                                {"evaluation": {f"cmprtr_{policy_name}": new_cmprtr}},
                                step=step,
                            )

        if (
            not FLAGS.debug
            and (step % FLAGS.save_interval == 0 or step == FLAGS.num_offline_steps)
        ):
            logging.info("Saving checkpoint...")
            checkpoint_path = None
            for policy_name in policy_to_agents.keys():
                checkpoint_path = checkpoints.save_checkpoint(
                    save_dir,
                    learning_policies[policy_name],
                    step=step,
                    keep=30,
                    prefix=f"{policy_name}_",
                )
            logging.info("Saved checkpoint to %s", checkpoint_path)

        timer.tock("total")

        if not FLAGS.debug and step % FLAGS.log_interval == 0:
            if update_info:
                wandb_logger.log({"training": jax.device_get(update_info)}, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)

        if (
            not FLAGS.debug
            and FLAGS.profile_code
            and step % FLAGS.profile_log_interval == 0
        ):
            profiling_info = profiler.summary(reset=True)
            if profiling_info:
                wandb_logger.log({"profiling": profiling_info}, step=step)
                logging.info("Profiling: %s", profiling_info)


if __name__ == "__main__":
    app.run(main)
