from ml_collections import ConfigDict

from experiments.configs import wsrl_config


def get_config(updates=None):
    config = wsrl_config.get_config()
     # Curriculum / guide settings
    config.n_curriculum_stages = 10
    config.guide_policy_path = ""
    config.eval_interval = 10000
    config.online_sampling_method = "mixed"
    config.offline_data_ratio = 0.5

   
    if updates is not None:
        config.update(ConfigDict(updates).copy_and_resolve_references())
    return config
