from experiments.configs.wsrl_config import get_config as get_wsrl_config
from ml_collections import ConfigDict

"""
GRL config shim — reuse the WSRL config for GRL experiments.
This keeps experiment names and defaults consistent while providing
a separate import path for future GRL-specific tweaks.
"""


def get_config(updates=None):
    """Return a copy of the WSRL config. `updates` is forwarded.

    Args:
        updates: optional dict of config updates passed to the underlying
                 WSRL get_config function.
    """
    config = get_wsrl_config(updates)
    config.n_curriculum_stages = 10
    config.guide_policy_path = ""
    config.eval_interval = 10000
    config.online_sampling_method = "mixed"
    config.offline_data_ratio = 0.5
    if updates is not None:
        config.update(ConfigDict(updates).copy_and_resolve_references())
    return config
