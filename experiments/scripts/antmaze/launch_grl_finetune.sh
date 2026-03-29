export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

python3 finetune.py \
--agent grl \
--config experiments/configs/train_config.py:antmaze_grl \
--project method-section \
--reward_scale 10.0 \
--reward_bias -5.0 \
--env antmaze-large-diverse-v2 \
--utd 4 \
--batch_size 1024 \
--warmup_steps 5000 \
--online_sampling_method mixed \
--offline_data_ratio 0 \
$@
