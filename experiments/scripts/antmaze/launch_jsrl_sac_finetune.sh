export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

python finetune.py \
--agent jsrl_sac \
--config experiments/configs/train_config.py:antmaze_jsrl_sac \
--project method-section \
--reward_scale 10.0 \
--reward_bias -5.0 \
--num_offline_steps 1_000_000 \
--n_eval_trajs 100 \
--env antmaze-large-diverse-v2 \
--utd 4 \
--batch_size 1024 \
--warmup_steps 5000 \
$@
