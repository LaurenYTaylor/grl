import os
import sys
from functools import partial
from typing import Optional

import chex
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from flax import core as flax_core
from flax.training import checkpoints
from absl import logging

from wsrl.agents.sac import SACAgent
from wsrl.common.common import JaxRLTrainState, nonpytree_field
from wsrl.common.typing import Params, PRNGKey, Data


class GRLSACAgent(SACAgent):
    """CalQL agent extended with a frozen guide policy and evaluation callback
    similar to the GRL agent. The guide policy is loaded from a checkpoint
    and stored as a FrozenDict on `guide_policy` (non-pytree)."""

    guide_policy: Optional[Params] = nonpytree_field(init=True, default=None)
    # Time step at which to start using the learner policy exclusively.
    cmprtr: float = jax.numpy.inf
    # Learning rate for updating cmprtr based on eval loss (tunable).
    cmprtr_lr: float = 0.1
    # Best evaluation score observed so far.
    best_eval_score: Optional[float] = None
    # The initial best evaluation score (set on first evaluation). Used as the
    # reference point when computing the loss = (avg_return - initial_best_eval_score)^2
    # for tuning `cmprtr`.
    initial_best_eval_score: Optional[float] = None
    # Decay factor applied to cmprtr when eval improves (kept for backwards
    # compatibility; not required by the loss-based update).
    decay_factor: float = 0.0
    # Number of curriculum stages (kept for compatibility with configs).
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

    def eval_callback(self, avg_return: float, avg_length: int) -> "GRLSACAgent":
        """
        Combined avg_length-based and loss-based tuning for `cmprtr`.

        Behavior:
        - On first evaluation: set `initial_best_eval_score`, `best_eval_score`,
            set `cmprtr` := `avg_length`, and initialize `decay_factor` :=
            floor(avg_length / n_curriculum_stages).
        - On every evaluation: compute diff = avg_return - initial_best_eval_score,
            loss = diff^2 and a gradient estimate g = 2*diff. Apply a gradient-like
            update: cmprtr <- max(0, cmprtr - cmprtr_lr * g).
        - Additionally, if avg_return improves over `best_eval_score`, apply the
            multiplicative/step decay: cmprtr <- max(0, cmprtr - decay_factor)
            and update `best_eval_score`.

        The combination keeps the avg_length initialization and decay behavior
        while allowing a tunable, loss-driven adjustment each evaluation.
        """
        # First-time initialization: set reference scores and cmprtr based on avg_length
        if self.initial_best_eval_score is None:
                logging.info("Setting initial_best_eval_score = %f", float(avg_return))
                new_cmprtr = avg_length
                decay_factor = jax.numpy.floor(avg_length / self.n_curriculum_stages)
                return self.replace(
                        initial_best_eval_score=float(avg_return),
                        best_eval_score=float(avg_return),
                        cmprtr=new_cmprtr,
                        decay_factor=decay_factor,
                )

        # Compute the loss-based gradient estimate
        diff = float(avg_return) - float(self.initial_best_eval_score)
        loss = diff * diff
        grad_est = 2.0 * diff

        # Apply the gradient-like update to cmprtr
        grad_updated_cmprtr = float(max(0.0, float(self.cmprtr) - float(self.cmprtr_lr) * grad_est))

        # If we've improved over the best_eval_score, apply the decay step as well
        improved = (self.best_eval_score is None) or (avg_return >= self.best_eval_score)
        if improved:
            # Apply decay on top of the gradient update
            post_decay_cmprtr = float(max(0.0, grad_updated_cmprtr - float(self.decay_factor)))
            logging.info(
                    "Eval improved: avg_return=%.3f best->%.3f loss=%.6f grad_est=%.6f cmprtr: %.6f->%.6f",
                    float(avg_return),
                    float(self.initial_best_eval_score),
                    float(loss),
                    float(grad_est),
                    float(self.cmprtr),
                    float(post_decay_cmprtr),
            )
            return self.replace(cmprtr=post_decay_cmprtr, best_eval_score=float(avg_return))
        else:
            logging.info(
                    "Eval not improved: avg_return=%.3f initial_best=%.3f loss=%.6f grad_est=%.6f cmprtr: %.6f->%.6f",
                    float(avg_return),
                    float(self.initial_best_eval_score),
                    float(loss),
                    float(grad_est),
                    float(self.cmprtr),
                    float(grad_updated_cmprtr),
            )
            return self.replace(cmprtr=grad_updated_cmprtr)

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