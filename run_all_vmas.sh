mkdir -p logs
export CUDA_VISIBLE_DEVICES=1
nohup bash experiments/scripts/vmas/launch_rlpd.sh --use_redq --env balance --profile_code False --num_envs 300 --eval_interval 2500 --log_interval 2500 --num_online_steps 40000 > logs/vmas_balance.log 2>&1 &
wait $!
nohup bash experiments/scripts/vmas/launch_rlpd.sh --use_redq --env discovery --profile_code False --num_envs 300 --eval_interval 2500 --log_interval 2500 --num_online_steps 40000 > logs/vmas_discovery.log 2>&1 &
wait $!
nohup bash experiments/scripts/vmas/launch_rlpd.sh --use_redq --env dispersion --profile_code False --num_envs 300 --eval_interval 2500 --log_interval 2500 --num_online_steps 40000 --replay_buffer_capacity 500000 > logs/vmas_dispersion.log 2>&1 &
wait $!
nohup bash experiments/scripts/vmas/launch_rlpd.sh --use_redq --env dropout --profile_code False --num_envs 300 --eval_interval 2500 --log_interval 2500 --num_online_steps 40000 > logs/vmas_dropout.log 2>&1 &
wait $!
nohup bash experiments/scripts/vmas/launch_rlpd.sh --use_redq --env flocking --profile_code False --num_envs 300 --eval_interval 2500 --log_interval 2500 --num_online_steps 40000 > logs/vmas_flocking.log 2>&1 &
wait $!
nohup bash experiments/scripts/vmas/launch_rlpd.sh --use_redq --env give_way --profile_code False --num_envs 300 --eval_interval 2500 --log_interval 2500 --num_online_steps 40000 > logs/vmas_give_way.log 2>&1 &
wait $!
nohup bash experiments/scripts/vmas/launch_rlpd.sh --use_redq --env navigation --profile_code False --num_envs 300 --eval_interval 2500 --log_interval 2500 --num_online_steps 40000 > logs/vmas_navigation.log 2>&1 &
wait $!
nohup bash experiments/scripts/vmas/launch_rlpd.sh --use_redq --env passage --profile_code False --num_envs 300 --eval_interval 2500 --log_interval 2500 --num_online_steps 40000 > logs/vmas_passage.log 2>&1 &
wait $!
nohup bash experiments/scripts/vmas/launch_rlpd.sh --use_redq --env reverse_transport --profile_code False --num_envs 300 --eval_interval 2500 --log_interval 2500 --num_online_steps 40000 > logs/vmas_reverse_transport.log 2>&1 &
wait $!
nohup bash experiments/scripts/vmas/launch_rlpd.sh --use_redq --env transport --profile_code False --num_envs 300 --eval_interval 2500 --log_interval 2500 --num_online_steps 40000 > logs/vmas_transport.log 2>&1 &
wait $!
nohup bash experiments/scripts/vmas/launch_rlpd.sh --use_redq --env wheel --profile_code False --num_envs 300 --eval_interval 2500 --log_interval 2500 --num_online_steps 40000 > logs/vmas_wheel.log 2>&1 &
wait $!