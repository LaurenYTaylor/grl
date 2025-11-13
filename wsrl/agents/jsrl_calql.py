import copy
from functools import partial
from typing import Optional, Tuple

import chex
import flax
import flax.linen as nn
from flax import core as flax_core
from flax.training import checkpoints

import jax
import jax.numpy as jnp
from ml_collections import ConfigDict
from overrides import overrides
import logging
import sys
import os
from wsrl.agents.sac import SACAgent
from wsrl.common.typing import Batch, Data, Params, PRNGKey
from wsrl.common.common import nonpytree_field


class JSRLCalQLAgent(SACAgent):
    """CalQL agent extended with a frozen guide policy and evaluation callback
    similar to the JSRL agent. The guide policy is loaded from a checkpoint
    and stored as a FrozenDict on `guide_policy` (non-pytree)."""

    guide_policy: Optional[Params] = nonpytree_field(init=True, default=None)
    # Time step at which to start using the learner policy exclusively.
    cmprtr: float = jax.numpy.inf
    # Best evaluation score observed for decaying cmprtr (optional).
    best_eval_score: Optional[float] = None
    # Decay factor applied to cmprtr when eval improves.
    decay_factor: int = 0
    # Number of curriculum stages.
    n_curriculum_stages: int = 10

    @partial(jax.jit, static_argnames=("argmax",))
    def sample_actions(
        self,
        observations: Data,
        env_step: int,
        *,
        seed: Optional[PRNGKey] = None,
        argmax: bool = False,
    ) -> jnp.ndarray:
        # Build PRNGKey
        key = jax.random.PRNGKey(seed[0] if seed is not None else 0)
        sel_key, sample_key = jax.random.split(key)
        operand = (observations, sample_key)

        def guide_branch(args):
            obs, rng = args
            # Use stored guide params if present
            if self.guide_policy is None:
                dist = self.forward_policy(obs, rng, grad_params=None, train=False)
            else:
                dist = self.forward_policy(obs, rng, grad_params=self.guide_policy, train=False)
            if argmax:
                return dist.mode()
            else:
                return dist.sample(seed=rng)

        def learner_branch(args):
            obs, rng = args
            dist = self.forward_policy(obs, rng, grad_params=None, train=False)
            if argmax:
                return dist.mode()
            else:
                return dist.sample(seed=rng)

        actions = jax.lax.cond(env_step > self.cmprtr, learner_branch, guide_branch, operand)
        return actions
    
    def eval_callback(self, avg_return: float, avg_length: int) -> "JSRLCalQLAgent":
        """
        Called after an evaluation. If the provided avg_return improves upon
        the stored `best_eval_score`, update `best_eval_score` and decay
        `cmprtr` multiplicatively by `decay_factor`. Returns a new agent
        instance (PyTree) with updated fields.
        """
        # If no previous score, treat this as an improvement.
        
        if (self.best_eval_score is None) or (avg_return >= self.best_eval_score):
            new_best = float(avg_return)
            if (self.best_eval_score is None):
                new_cmprtr = avg_length
                decay_factor = jax.numpy.floor(avg_length/self.n_curriculum_stages)
                return self.replace(cmprtr=new_cmprtr, best_eval_score=new_best, decay_factor=decay_factor)
            else:
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
                # Remove critic/value if present
                mutable_params.pop("modules_critic", None)
                mutable_params.pop("modules_value", None)


                # As a safety, ensure the actor exists
                if "modules_actor" not in mutable_params:
                    logging.error(
                        "No actor params found in checkpoint %s (looked for 'modules_actor' or 'actor').",
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

        # require guide_policy by default for JSRL-CalQL
        assert guide_policy is not None, "a guide_policy must be provided through the guide_policy_path kwarg"

        kwargs["use_calql"] = True
        kwargs["n_curriculum_stages"] = n_curriculum_stages
        # Remove from kwargs if present, to avoid double passing
        kwargs.pop("guide_policy_path", None)
        agent = super().create(*args, **kwargs)
        # Use .replace to set non-pytree fields
        agent = agent.replace(guide_policy=guide_policy, n_curriculum_stages=n_curriculum_stages)
        return agent