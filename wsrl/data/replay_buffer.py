import os
from typing import Iterable, Optional, Union

import gym
import gym.spaces
import jax
import numpy as np
import torch
from absl import flags

from wsrl.data.dataset import Dataset, DatasetDict, _sample
from wsrl.envs.env_common import (
    _determine_whether_sparse_reward,
    _get_negative_reward,
    calc_return_to_go,
)


def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert (
            dataset_dict.keys() == data_dict.keys()
        ), f"{dataset_dict.keys()} != {data_dict.keys()}"
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


def _get_batch_length(data: Union[np.ndarray, DatasetDict]) -> int:
    if isinstance(data, np.ndarray):
        return len(data)
    if torch.is_tensor(data):
        return len(data)
    if isinstance(data, dict):
        batch_len = None
        for value in data.values():
            value_batch_len = _get_batch_length(value)
            if batch_len is None:
                batch_len = value_batch_len
            else:
                assert (
                    batch_len == value_batch_len
                ), "Inconsistent batch lengths in inserted replay data."
        if batch_len is None:
            raise ValueError("Cannot infer batch length from empty dictionary.")
        return batch_len
    raise TypeError()


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        seed: Optional[int] = None,
        discount: Optional[float] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=bool),
            dones=np.empty((capacity,), dtype=np.float32),
            ts=np.empty((capacity,), dtype=np.int32),
        )

        super().__init__(dataset_dict, seed)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0
        self._sequential_index = 0
        self.unsampled_indices = list(range(self._size))
        self._discount = discount

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def _get_insert_indices(self, batch_size: int) -> np.ndarray:
        return (self._insert_index + np.arange(batch_size)) % self._capacity

    def sample_without_repeat(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
    ) -> dict:
        if keys is None:
            keys = self.dataset_dict.keys()

        batch = dict()
        if len(self.unsampled_indices) < batch_size:
            raise ValueError("Not enough samples left to sample without repeat.")
        selected_indices = []
        for _ in range(batch_size):
            idx = self.np_random.randint(len(self.unsampled_indices))
            selected_indices.append(self.unsampled_indices[idx])
            # Swap the selected index with the last unselected index
            self.unsampled_indices[idx], self.unsampled_indices[-1] = (
                self.unsampled_indices[-1],
                self.unsampled_indices[idx],
            )
            # Remove the last unselected index (which is now the selected index)
            self.unsampled_indices.pop()

        for k in keys:
            batch[k] = _sample(self.dataset_dict[k], np.array(selected_indices))

        return batch

    def save(self, save_dir):
        save_buffer_file = os.path.join(save_dir, "online_buffer.npy")
        save_size_file = os.path.join(save_dir, "size.npy")
        np.save(save_buffer_file, self.dataset_dict)
        np.save(save_size_file, self._size)

    def load(self, save_dir):
        # TODO: maybe make sure the dataset_dict thats being loaded has mc_returns if self is ReplayBufferMC
        save_buffer_file = os.path.join(save_dir, "online_buffer.npy")
        save_size_file = os.path.join(save_dir, "size.npy")
        self.dataset_dict = np.load(save_buffer_file, allow_pickle=True).item()
        self._size = np.load(save_size_file, allow_pickle=True).item()
        self.unsampled_indices = list(range(self._size))


class ReplayBufferMC(ReplayBuffer):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        seed: Optional[int] = None,
        discount: Optional[float] = None,
    ):
        assert discount is not None, "ReplayBufferMC requires a discount factor"
        super().__init__(
            observation_space,
            action_space,
            capacity,
            next_observation_space,
            seed,
            discount,
        )

        mc_returns = np.empty((capacity,), dtype=np.float32)
        self.dataset_dict["mc_returns"] = mc_returns

        self._allow_idxs = []
        self._traj_start_idx = 0

    def insert(self, data_dict: DatasetDict):
        # assumes replay buffer capacity is more than the number of online steps
        assert self._size < self._capacity, "replay buffer has reached capacity"

        data_dict["mc_returns"] = None
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        # if "dones" not in data_dict:
        #     data_dict["dones"] = 1 - data_dict["masks"]

        if data_dict["dones"] == 1.0:
            # compute the mc_returns
            FLAGS = flags.FLAGS
            rewards = self.dataset_dict["rewards"][
                self._traj_start_idx : self._insert_index + 1
            ]
            masks = self.dataset_dict["masks"][
                self._traj_start_idx : self._insert_index + 1
            ]
            self.dataset_dict["mc_returns"][
                self._traj_start_idx : self._insert_index + 1
            ] = calc_return_to_go(
                FLAGS.env,
                rewards,
                masks,
                self._discount,
            )

            self._allow_idxs.extend(
                list(range(self._traj_start_idx, self._insert_index + 1))
            )
            self._traj_start_idx = self._insert_index + 1

        self._size += 1
        self._insert_index += 1

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> dict:
        if indx is None:
            indx = self.np_random.choice(
                self._allow_idxs, size=batch_size, replace=True
            )
        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()

        for k in keys:
            batch[k] = _sample(self.dataset_dict[k], indx)

        return batch


class ParallelReplayBuffer(ReplayBuffer):
    """Replay buffer that supports batched insertion from parallel environments."""

    def insert_batch(self, data_dict: DatasetDict):
        batch_size = _get_batch_length(data_dict)
        insert_indices = self._get_insert_indices(batch_size)
        _insert_recursively(self.dataset_dict, data_dict, insert_indices)

        self._insert_index = (self._insert_index + batch_size) % self._capacity
        self._size = min(self._size + batch_size, self._capacity)
        self.unsampled_indices = list(range(self._size))


class ParallelReplayBufferMC(ReplayBufferMC):
    """MC replay buffer with batched insertion from parallel environments."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        seed: Optional[int] = None,
        discount: Optional[float] = None,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            capacity=capacity,
            next_observation_space=next_observation_space,
            seed=seed,
            discount=discount,
        )
        self._traj_indices_per_env = None

    def insert_batch(self, data_dict: DatasetDict):
        batch_size = _get_batch_length(data_dict)
        assert (
            self._size + batch_size <= self._capacity
        ), "replay buffer has reached capacity"

        if self._traj_indices_per_env is None:
            self._traj_indices_per_env = [[] for _ in range(batch_size)]
        else:
            assert len(self._traj_indices_per_env) == batch_size, (
                f"Expected {len(self._traj_indices_per_env)} parallel envs, "
                f"got {batch_size}"
            )

        batch_to_insert = dict(data_dict)
        batch_to_insert["mc_returns"] = np.empty((batch_size,), dtype=np.float32)

        insert_indices = self._get_insert_indices(batch_size)
        _insert_recursively(self.dataset_dict, batch_to_insert, insert_indices)

        dones = np.asarray(data_dict["dones"]).astype(bool)
        for env_idx, replay_idx in enumerate(insert_indices):
            self._traj_indices_per_env[env_idx].append(int(replay_idx))
            if dones[env_idx]:
                traj_indices = np.asarray(
                    self._traj_indices_per_env[env_idx], dtype=np.int32
                )
                rewards = self.dataset_dict["rewards"][traj_indices]
                masks = self.dataset_dict["masks"][traj_indices]
                self.dataset_dict["mc_returns"][traj_indices] = calc_return_to_go(
                    flags.FLAGS.env,
                    rewards,
                    masks,
                    self._discount,
                )
                self._allow_idxs.extend(traj_indices.tolist())
                self._traj_indices_per_env[env_idx] = []

        self._size += batch_size
        self._insert_index += batch_size


def _numpy_dtype_to_torch(np_dtype) -> torch.dtype:
    return torch.from_numpy(np.empty((), dtype=np_dtype)).dtype


def _init_torch_replay_dict(
    obs_space: gym.Space, capacity: int, device: Union[str, torch.device]
):
    if isinstance(obs_space, gym.spaces.Box):
        return torch.empty(
            (capacity, *obs_space.shape),
            dtype=_numpy_dtype_to_torch(obs_space.dtype),
            device=device,
        )
    if isinstance(obs_space, gym.spaces.Dict):
        return {
            key: _init_torch_replay_dict(value, capacity, device)
            for key, value in obs_space.spaces.items()
        }
    raise TypeError()


def _normalize_torch_indices(indices, device: torch.device):
    if isinstance(indices, int):
        return indices
    if torch.is_tensor(indices):
        return indices.to(device=device, dtype=torch.long)
    return torch.as_tensor(indices, device=device, dtype=torch.long)


def _to_torch_tree(data, reference):
    if torch.is_tensor(reference):
        if torch.is_tensor(data):
            return data.to(device=reference.device, dtype=reference.dtype)
        return torch.as_tensor(data, device=reference.device, dtype=reference.dtype)
    if isinstance(reference, dict):
        assert reference.keys() == data.keys(), f"{reference.keys()} != {data.keys()}"
        return {
            key: _to_torch_tree(data[key], reference[key]) for key in reference.keys()
        }
    raise TypeError()


def _insert_torch_recursively(dataset_dict, data_dict, insert_index):
    if torch.is_tensor(dataset_dict):
        normalized_index = _normalize_torch_indices(insert_index, dataset_dict.device)
        dataset_dict[normalized_index] = _to_torch_tree(data_dict, dataset_dict[normalized_index])
        return
    if isinstance(dataset_dict, dict):
        assert dataset_dict.keys() == data_dict.keys(), f"{dataset_dict.keys()} != {data_dict.keys()}"
        for key in dataset_dict.keys():
            _insert_torch_recursively(dataset_dict[key], data_dict[key], insert_index)
        return
    raise TypeError()


def _sample_torch(dataset_dict, indx):
    if torch.is_tensor(dataset_dict):
        normalized_index = _normalize_torch_indices(indx, dataset_dict.device)
        return dataset_dict[normalized_index]
    if isinstance(dataset_dict, dict):
        return {key: _sample_torch(value, indx) for key, value in dataset_dict.items()}
    raise TypeError("Unsupported type.")


def _torch_to_jax(data):
    if torch.is_tensor(data):
        return jax.dlpack.from_dlpack(data.contiguous())
    if isinstance(data, dict):
        return {key: _torch_to_jax(value) for key, value in data.items()}
    raise TypeError("Unsupported type.")


def _torch_calc_return_to_go(
    env_name,
    rewards: torch.Tensor,
    masks: torch.Tensor,
    gamma: float,
    reward_scale=None,
    reward_bias=None,
    infinite_horizon: bool = False,
):
    if rewards.numel() == 0:
        return torch.empty_like(rewards, dtype=torch.float32)

    if reward_scale is None or reward_bias is None:
        assert reward_scale is None and reward_bias is None
        reward_scale = flags.FLAGS.reward_scale
        reward_bias = flags.FLAGS.reward_bias

    try:
        is_sparse_reward = _determine_whether_sparse_reward(env_name)
    except NotImplementedError:
        is_sparse_reward = False

    if is_sparse_reward:
        reward_neg = _get_negative_reward(env_name, reward_scale, reward_bias)
        reward_neg_tensor = torch.as_tensor(
            reward_neg, dtype=torch.float32, device=rewards.device
        )
        if torch.all(rewards == reward_neg_tensor):
            return torch.full_like(
                rewards,
                float(reward_neg / (1 - gamma)),
                dtype=torch.float32,
                device=rewards.device,
            )

    return_to_go = torch.empty_like(rewards, dtype=torch.float32)
    prev_return = torch.tensor(0.0, dtype=torch.float32, device=rewards.device)
    if infinite_horizon:
        prev_return = rewards[-1] / (1 - gamma)
    for idx in range(rewards.shape[0] - 1, -1, -1):
        prev_return = rewards[idx] + gamma * prev_return * masks[idx].to(torch.float32)
        return_to_go[idx] = prev_return
    return return_to_go


class TorchGPUReplayBuffer:
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        seed: Optional[int] = None,
        discount: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space
        self.device = torch.device(device or "cuda")
        self._np_random = np.random.RandomState(seed)
        self._discount = discount
        self._capacity = capacity
        self._size = 0
        self._insert_index = 0

        observation_data = _init_torch_replay_dict(observation_space, capacity, self.device)
        next_observation_data = _init_torch_replay_dict(
            next_observation_space, capacity, self.device
        )
        action_shape = getattr(action_space, "shape", ())
        action_dtype = _numpy_dtype_to_torch(action_space.dtype)
        self.dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=torch.empty((capacity, *action_shape), dtype=action_dtype, device=self.device),
            rewards=torch.empty((capacity,), dtype=torch.float32, device=self.device),
            masks=torch.empty((capacity,), dtype=torch.bool, device=self.device),
            dones=torch.empty((capacity,), dtype=torch.float32, device=self.device),
            ts=torch.empty((capacity,), dtype=torch.int32, device=self.device),
        )

    def __len__(self) -> int:
        return self._size

    def _get_insert_indices(self, batch_size: int):
        return (self._insert_index + np.arange(batch_size)) % self._capacity

    def insert(self, data_dict: DatasetDict):
        _insert_torch_recursively(self.dataset_dict, data_dict, self._insert_index)
        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def insert_batch(self, data_dict: DatasetDict):
        batch_size = _get_batch_length(data_dict)
        insert_indices = self._get_insert_indices(batch_size)
        _insert_torch_recursively(self.dataset_dict, data_dict, insert_indices)
        self._insert_index = (self._insert_index + batch_size) % self._capacity
        self._size = min(self._size + batch_size, self._capacity)

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> dict:
        if indx is None:
            indx = self._np_random.choice(self._size, size=batch_size, replace=True)
        if keys is None:
            keys = self.dataset_dict.keys()
        batch = {key: _sample_torch(self.dataset_dict[key], indx) for key in keys}
        return _torch_to_jax(batch)

    def save(self, save_dir):
        torch.save(
            {
                "dataset_dict": self.dataset_dict,
                "size": self._size,
                "insert_index": self._insert_index,
            },
            os.path.join(save_dir, "online_buffer.pt"),
        )

    def load(self, save_dir):
        payload = torch.load(
            os.path.join(save_dir, "online_buffer.pt"),
            map_location=self.device,
        )
        self.dataset_dict = payload["dataset_dict"]
        self._size = payload["size"]
        self._insert_index = payload["insert_index"]


class TorchGPUReplayBufferMC(TorchGPUReplayBuffer):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        seed: Optional[int] = None,
        discount: Optional[float] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        assert discount is not None, "TorchGPUReplayBufferMC requires a discount factor"
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            capacity=capacity,
            next_observation_space=next_observation_space,
            seed=seed,
            discount=discount,
            device=device,
        )
        self.dataset_dict["mc_returns"] = torch.empty(
            (capacity,), dtype=torch.float32, device=self.device
        )
        self._allow_idxs = []
        self._traj_indices_per_env = None

    def insert_batch(self, data_dict: DatasetDict):
        batch_size = _get_batch_length(data_dict)
        assert self._size + batch_size <= self._capacity, "replay buffer has reached capacity"

        if self._traj_indices_per_env is None:
            self._traj_indices_per_env = [[] for _ in range(batch_size)]
        else:
            assert len(self._traj_indices_per_env) == batch_size, (
                f"Expected {len(self._traj_indices_per_env)} parallel envs, got {batch_size}"
            )

        insert_indices = self._get_insert_indices(batch_size)
        batch_to_insert = dict(data_dict)
        batch_to_insert["mc_returns"] = torch.empty(
            (batch_size,), dtype=torch.float32, device=self.device
        )
        _insert_torch_recursively(self.dataset_dict, batch_to_insert, insert_indices)

        dones = _to_torch_tree(data_dict["dones"], self.dataset_dict["dones"])
        for env_idx, replay_idx in enumerate(insert_indices):
            self._traj_indices_per_env[env_idx].append(int(replay_idx))
            if bool(dones[env_idx].item()):
                traj_indices = np.asarray(self._traj_indices_per_env[env_idx], dtype=np.int32)
                rewards = self.dataset_dict["rewards"][traj_indices]
                masks = self.dataset_dict["masks"][traj_indices]
                mc_returns = _torch_calc_return_to_go(
                    flags.FLAGS.env,
                    rewards,
                    masks,
                    self._discount,
                )
                self.dataset_dict["mc_returns"][traj_indices] = mc_returns
                self._allow_idxs.extend(traj_indices.tolist())
                self._traj_indices_per_env[env_idx] = []

        self._size += batch_size
        self._insert_index = (self._insert_index + batch_size) % self._capacity

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> dict:
        if indx is None:
            indx = self._np_random.choice(self._allow_idxs, size=batch_size, replace=True)
        if keys is None:
            keys = self.dataset_dict.keys()
        batch = {key: _sample_torch(self.dataset_dict[key], indx) for key in keys}
        return _torch_to_jax(batch)
