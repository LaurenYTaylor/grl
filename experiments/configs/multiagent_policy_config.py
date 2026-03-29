from ml_collections import ConfigDict


def _shared_config():
    return ConfigDict(
        dict(
            policy_sharing="shared",
            default_policy="shared_policy",
            agent_policy_mapping=[],
        )
    )


def _independent_config():
    return ConfigDict(
        dict(
            policy_sharing="independent",
            default_policy=None,
            agent_policy_mapping=[],
        )
    )


def get_config(config_string=None):
    """Policy-sharing config for multiagent training.

    `agent_policy_mapping` is a list of single-entry dictionaries like:
    [{"agent_0": "policy_0"}, {"agent_1": "policy_1"}]

    Any agent not explicitly listed uses `default_policy`.

    `config_string` can be either:
    - a preset name: `shared` or `independent`
    - an environment name: e.g. `waterfall`

    Environment names are looked up in `environment_configs`. Unknown env names
    fall back to the shared-policy default.
    """
    possible_structures = {
        "shared": _shared_config(),
        "independent": _independent_config(),
    }

    environment_configs = {
        # VMAS defaults
        "waterfall": _shared_config(),
        "dispersion": _independent_config(),
        "give_way": _independent_config(),
        # Add more env-specific overrides here as needed.
        # Example:
        # "some_env": ConfigDict(
        #     dict(
        #         policy_sharing="shared",
        #         default_policy="shared_policy",
        #         agent_policy_mapping=[
        #             {"agent_0": "policy_a"},
        #             {"agent_1": "policy_a"},
        #             {"agent_2": "policy_b"},
        #         ],
        #     )
        # ),
    }

    if config_string is None:
        config_string = "shared"

    if config_string in possible_structures:
        return possible_structures[config_string]

    if config_string in environment_configs:
        return environment_configs[config_string]

    return _shared_config()
