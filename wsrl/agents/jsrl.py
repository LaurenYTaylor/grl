import copy
from functools import partial
from typing import Optional

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict
from flax.training import checkpoints
from flax import core as flax_core
from absl import logging
import os
import sys

from wsrl.common.common import JaxRLTrainState, FrozenTrainState, ModuleDict, nonpytree_field
from wsrl.common.optimizers import make_optimizer
from wsrl.common.typing import Batch, Data, Params, PRNGKey
from wsrl.networks.actor_critic_nets import Critic, Policy, ValueCritic, ensemblize
from wsrl.networks.mlp import MLP
from wsrl.agents.iql import IQLAgent

class JSRLAgent(IQLAgent):
    guide_policy: Optional[Params] = nonpytree_field(init=True, default=None)
    n_curriculum_stages: int = 10
    cmprtr: float = jax.numpy.inf
    best_eval_score: Optional[float] = None
    decay_factor: int = 0
    

    @partial(jax.jit, static_argnames=("argmax",))
    def sample_actions(
        self,
        observations: Data,
        env_step: int,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
    ) -> jnp.ndarray:
        operand = (observations, seed)

        def guide_branch(args):
            obs, rng = args
            if self.guide_policy is None:
                dist = self.forward_policy(obs, rng, grad_params=None, train=False)
            else:
                dist = self.forward_policy(obs, rng, grad_params=self.guide_policy, train=False)
            return dist.mode()

        def learner_branch(args):
            obs, rng = args
            dist = self.forward_policy(obs, rng, grad_params=None, train=False)
            if argmax:
                return dist.mode()
            else:
                return dist.sample(seed=rng)

        actions = jax.lax.cond(env_step > self.cmprtr, learner_branch, guide_branch, operand)
        return actions

    def eval_callback(self, avg_return: float, avg_length: int) -> "JSRLAgent":
        if (self.best_eval_score is None) or (avg_return >= 0.95*self.best_eval_score):
            if (self.best_eval_score is None):
                new_best = float(avg_return)
                if (self.n_curriculum_stages <= 0):
                    new_cmprtr = 0
                    decay_factor = 0
                else:
                    new_cmprtr = avg_length
                    decay_factor = jax.numpy.floor(avg_length/self.n_curriculum_stages)
                return self.replace(cmprtr=new_cmprtr, best_eval_score=new_best, decay_factor=decay_factor)
            else:
                new_best = max(float(avg_return), self.best_eval_score)
                new_cmprtr = float(max(0.0, self.cmprtr - self.decay_factor))
                return self.replace(cmprtr=new_cmprtr, best_eval_score=new_best)
        return self
    
    @classmethod
    def create(cls, *args, **kwargs):
        guide_policy = None
        guide_policy_path = kwargs.get("guide_policy_path", "")
        n_curriculum_stages = kwargs.get("n_curriculum_stages", 10)
        if guide_policy_path:
            if os.path.exists(guide_policy_path):
                try:
                    ckpt = checkpoints.restore_checkpoint(guide_policy_path, target=None)
                except Exception as e:
                    logging.error("Failed to restore checkpoint %s: %s", guide_policy_path, e)
                    sys.exit(1)

                params = ckpt.get("state", {}).get("params") or ckpt.get("params") or None
                if params is None:
                    logging.error("No params found in checkpoint %s", guide_policy_path)
                    sys.exit(1)

                mutable_params = flax_core.unfreeze(params)
                mutable_params.pop("modules_critic", None)
                mutable_params.pop("modules_value", None)
                if "modules_actor" not in mutable_params:
                    logging.error(
                        "No actor params found in checkpoint %s (looked for 'modules_actor').",
                        guide_policy_path,
                    )
                    sys.exit(1)
                guide_policy = flax_core.freeze(mutable_params)
                logging.info(
                    "Loaded params from checkpoint %s into agent.guide_policy (critic/value removed where present).",
                    guide_policy_path,
                )
            else:
                logging.error("guide_policy_path specified but file not found: %s", guide_policy_path)
                sys.exit(1)

        if n_curriculum_stages > 0:
            assert guide_policy is not None, "a guide_policy must be provided through the guide_policy_path kwarg"

        kwargs["n_curriculum_stages"] = n_curriculum_stages
        kwargs.pop("guide_policy_path", None)
        agent = super(JSRLAgent, cls).create(*args, **kwargs)
        agent = agent.replace(guide_policy=guide_policy, n_curriculum_stages=n_curriculum_stages)
        return agent