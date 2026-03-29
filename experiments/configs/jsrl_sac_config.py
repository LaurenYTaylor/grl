from ml_collections import ConfigDict

from experiments.configs import wsrl_config


def get_config(updates=None):
    config = wsrl_config.get_config()
     # Curriculum / guide settings
    config.n_curriculum_stages = 0
    config.guide_policy_path = ""
    config.eval_interval = 5000
    config.online_sampling_method = "mixed"
    config.offline_data_ratio = 0
    config.guided_rl = True
    #config.bc_loss_weight = 0.1

   
    if updates is not None:
        config.update(ConfigDict(updates).copy_and_resolve_references())
    return config
