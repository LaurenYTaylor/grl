#!/usr/bin/env bash
# Generated run commands for antmaze, adroit and locomotion envs × ALGS
# Excludes all calql × locomotion combos (per request).
# Run from repository root: bash generated_run_commands.sh
set -euo pipefail

# ----------------
# Antmaze
# ----------------
# antmaze × calql
bash experiments/scripts/antmaze/launch_calql_finetune.sh --use_redq --env antmaze-umaze-v2
bash experiments/scripts/antmaze/launch_calql_finetune.sh --use_redq --env antmaze-umaze-diverse-v2
bash experiments/scripts/antmaze/launch_calql_finetune.sh --use_redq --env antmaze-medium-play-v2
bash experiments/scripts/antmaze/launch_calql_finetune.sh --use_redq --env antmaze-medium-diverse-v2
bash experiments/scripts/antmaze/launch_calql_finetune.sh --use_redq --env antmaze-large-play-v2
bash experiments/scripts/antmaze/launch_calql_finetune.sh --use_redq --env antmaze-large-diverse-v2
bash experiments/scripts/antmaze/launch_calql_finetune.sh --use_redq --env antmaze-ultra-play-v2
bash experiments/scripts/antmaze/launch_calql_finetune.sh --use_redq --env antmaze-ultra-diverse-v2

# antmaze × cql
bash experiments/scripts/antmaze/launch_cql_finetune.sh --use_redq --env antmaze-umaze-v2
bash experiments/scripts/antmaze/launch_cql_finetune.sh --use_redq --env antmaze-umaze-diverse-v2
bash experiments/scripts/antmaze/launch_cql_finetune.sh --use_redq --env antmaze-medium-play-v2
bash experiments/scripts/antmaze/launch_cql_finetune.sh --use_redq --env antmaze-medium-diverse-v2
bash experiments/scripts/antmaze/launch_cql_finetune.sh --use_redq --env antmaze-large-play-v2
bash experiments/scripts/antmaze/launch_cql_finetune.sh --use_redq --env antmaze-large-diverse-v2
bash experiments/scripts/antmaze/launch_cql_finetune.sh --use_redq --env antmaze-ultra-play-v2
bash experiments/scripts/antmaze/launch_cql_finetune.sh --use_redq --env antmaze-ultra-diverse-v2

# antmaze × grl
bash experiments/scripts/antmaze/launch_grl_finetune.sh --use_redq --env antmaze-umaze-v2
bash experiments/scripts/antmaze/launch_grl_finetune.sh --use_redq --env antmaze-umaze-diverse-v2
bash experiments/scripts/antmaze/launch_grl_finetune.sh --use_redq --env antmaze-medium-play-v2
bash experiments/scripts/antmaze/launch_grl_finetune.sh --use_redq --env antmaze-medium-diverse-v2
bash experiments/scripts/antmaze/launch_grl_finetune.sh --use_redq --env antmaze-large-play-v2
bash experiments/scripts/antmaze/launch_grl_finetune.sh --use_redq --env antmaze-large-diverse-v2
bash experiments/scripts/antmaze/launch_grl_finetune.sh --use_redq --env antmaze-ultra-play-v2
bash experiments/scripts/antmaze/launch_grl_finetune.sh --use_redq --env antmaze-ultra-diverse-v2

# antmaze × iql
bash experiments/scripts/antmaze/launch_iql_finetune.sh --use_redq --env antmaze-umaze-v2
bash experiments/scripts/antmaze/launch_iql_finetune.sh --use_redq --env antmaze-umaze-diverse-v2
bash experiments/scripts/antmaze/launch_iql_finetune.sh --use_redq --env antmaze-medium-play-v2
bash experiments/scripts/antmaze/launch_iql_finetune.sh --use_redq --env antmaze-medium-diverse-v2
bash experiments/scripts/antmaze/launch_iql_finetune.sh --use_redq --env antmaze-large-play-v2
bash experiments/scripts/antmaze/launch_iql_finetune.sh --use_redq --env antmaze-large-diverse-v2
bash experiments/scripts/antmaze/launch_iql_finetune.sh --use_redq --env antmaze-ultra-play-v2
bash experiments/scripts/antmaze/launch_iql_finetune.sh --use_redq --env antmaze-ultra-diverse-v2

# antmaze × jsrl_random
bash experiments/scripts/antmaze/launch_jsrl_random_finetune.sh --use_redq --env antmaze-umaze-v2
bash experiments/scripts/antmaze/launch_jsrl_random_finetune.sh --use_redq --env antmaze-umaze-diverse-v2
bash experiments/scripts/antmaze/launch_jsrl_random_finetune.sh --use_redq --env antmaze-medium-play-v2
bash experiments/scripts/antmaze/launch_jsrl_random_finetune.sh --use_redq --env antmaze-medium-diverse-v2
bash experiments/scripts/antmaze/launch_jsrl_random_finetune.sh --use_redq --env antmaze-large-play-v2
bash experiments/scripts/antmaze/launch_jsrl_random_finetune.sh --use_redq --env antmaze-large-diverse-v2
bash experiments/scripts/antmaze/launch_jsrl_random_finetune.sh --use_redq --env antmaze-ultra-play-v2
bash experiments/scripts/antmaze/launch_jsrl_random_finetune.sh --use_redq --env antmaze-ultra-diverse-v2

# antmaze × jsrl
bash experiments/scripts/antmaze/launch_jsrl_finetune.sh --use_redq --env antmaze-umaze-v2
bash experiments/scripts/antmaze/launch_jsrl_finetune.sh --use_redq --env antmaze-umaze-diverse-v2
bash experiments/scripts/antmaze/launch_jsrl_finetune.sh --use_redq --env antmaze-medium-play-v2
bash experiments/scripts/antmaze/launch_jsrl_finetune.sh --use_redq --env antmaze-medium-diverse-v2
bash experiments/scripts/antmaze/launch_jsrl_finetune.sh --use_redq --env antmaze-large-play-v2
bash experiments/scripts/antmaze/launch_jsrl_finetune.sh --use_redq --env antmaze-large-diverse-v2
bash experiments/scripts/antmaze/launch_jsrl_finetune.sh --use_redq --env antmaze-ultra-play-v2
bash experiments/scripts/antmaze/launch_jsrl_finetune.sh --use_redq --env antmaze-ultra-diverse-v2

# antmaze × pex
bash experiments/scripts/antmaze/launch_pex_finetune.sh --use_redq --env antmaze-umaze-v2
bash experiments/scripts/antmaze/launch_pex_finetune.sh --use_redq --env antmaze-umaze-diverse-v2
bash experiments/scripts/antmaze/launch_pex_finetune.sh --use_redq --env antmaze-medium-play-v2
bash experiments/scripts/antmaze/launch_pex_finetune.sh --use_redq --env antmaze-medium-diverse-v2
bash experiments/scripts/antmaze/launch_pex_finetune.sh --use_redq --env antmaze-large-play-v2
bash experiments/scripts/antmaze/launch_pex_finetune.sh --use_redq --env antmaze-large-diverse-v2
bash experiments/scripts/antmaze/launch_pex_finetune.sh --use_redq --env antmaze-ultra-play-v2
bash experiments/scripts/antmaze/launch_pex_finetune.sh --use_redq --env antmaze-ultra-diverse-v2

# antmaze × sac (via wsrl script)
bash experiments/scripts/antmaze/launch_wsrl_finetune.sh --use_redq --env antmaze-umaze-v2
bash experiments/scripts/antmaze/launch_wsrl_finetune.sh --use_redq --env antmaze-umaze-diverse-v2
bash experiments/scripts/antmaze/launch_wsrl_finetune.sh --use_redq --env antmaze-medium-play-v2
bash experiments/scripts/antmaze/launch_wsrl_finetune.sh --use_redq --env antmaze-medium-diverse-v2
bash experiments/scripts/antmaze/launch_wsrl_finetune.sh --use_redq --env antmaze-large-play-v2
bash experiments/scripts/antmaze/launch_wsrl_finetune.sh --use_redq --env antmaze-large-diverse-v2
bash experiments/scripts/antmaze/launch_wsrl_finetune.sh --use_redq --env antmaze-ultra-play-v2
bash experiments/scripts/antmaze/launch_wsrl_finetune.sh --use_redq --env antmaze-ultra-diverse-v2

# ----------------
# Adroit
# ----------------
# adroit × calql
bash experiments/scripts/adroit/launch_calql_finetune.sh --use_redq --env pen-binary-v0
bash experiments/scripts/adroit/launch_calql_finetune.sh --use_redq --env door-binary-v0

# adroit × cql
bash experiments/scripts/adroit/launch_cql_finetune.sh --use_redq --env pen-binary-v0
bash experiments/scripts/adroit/launch_cql_finetune.sh --use_redq --env door-binary-v0

# adroit × iql
bash experiments/scripts/adroit/launch_iql_finetune.sh --use_redq --env pen-binary-v0
bash experiments/scripts/adroit/launch_iql_finetune.sh --use_redq --env door-binary-v0

# adroit × sac (via wsrl script)
bash experiments/scripts/adroit/launch_wsrl_finetune.sh --use_redq --env pen-binary-v0
bash experiments/scripts/adroit/launch_wsrl_finetune.sh --use_redq --env door-binary-v0

# ----------------
# Locomotion
# ----------------
# Note: calql × locomotion combos are intentionally excluded.

# locomotion × cql
bash experiments/scripts/locomotion/launch_cql_finetune.sh --use_redq --env halfcheetah-medium-replay-v2
bash experiments/scripts/locomotion/launch_cql_finetune.sh --use_redq --env halfcheetah-random-v2
bash experiments/scripts/locomotion/launch_cql_finetune.sh --use_redq --env halfcheetah-expert-v2
bash experiments/scripts/locomotion/launch_cql_finetune.sh --use_redq --env hopper-medium-replay-v2
bash experiments/scripts/locomotion/launch_cql_finetune.sh --use_redq --env hopper-random-v2
bash experiments/scripts/locomotion/launch_cql_finetune.sh --use_redq --env hopper-expert-v2
bash experiments/scripts/locomotion/launch_cql_finetune.sh --use_redq --env walker-medium-replay-v2
bash experiments/scripts/locomotion/launch_cql_finetune.sh --use_redq --env walker-random-v2
bash experiments/scripts/locomotion/launch_cql_finetune.sh --use_redq --env walker-expert-v2

# locomotion × iql
bash experiments/scripts/locomotion/launch_iql_finetune.sh --use_redq --env halfcheetah-medium-replay-v2
bash experiments/scripts/locomotion/launch_iql_finetune.sh --use_redq --env halfcheetah-random-v2
bash experiments/scripts/locomotion/launch_iql_finetune.sh --use_redq --env halfcheetah-expert-v2
bash experiments/scripts/locomotion/launch_iql_finetune.sh --use_redq --env hopper-medium-replay-v2
bash experiments/scripts/locomotion/launch_iql_finetune.sh --use_redq --env hopper-random-v2
bash experiments/scripts/locomotion/launch_iql_finetune.sh --use_redq --env hopper-expert-v2
bash experiments/scripts/locomotion/launch_iql_finetune.sh --use_redq --env walker-medium-replay-v2
bash experiments/scripts/locomotion/launch_iql_finetune.sh --use_redq --env walker-random-v2
bash experiments/scripts/locomotion/launch_iql_finetune.sh --use_redq --env walker-expert-v2

# locomotion × sac (via wsrl script)
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh --use_redq --env halfcheetah-medium-replay-v2
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh --use_redq --env halfcheetah-random-v2
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh --use_redq --env halfcheetah-expert-v2
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh --use_redq --env hopper-medium-replay-v2
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh --use_redq --env hopper-random-v2
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh --use_redq --env hopper-expert-v2
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh --use_redq --env walker-medium-replay-v2
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh --use_redq --env walker-random-v2
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh --use_redq --env walker-expert-v2

# End of generated commands

