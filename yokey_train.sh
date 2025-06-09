USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.


python train.py --config-name _7_alfworld trainer.experiment_name=alfworld_grpo_v2 algorithm.adv_estimator=grpo \
agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True

python train.py --config-name _7_alfworld trainer.experiment_name=alfworld_ppo $USE_PPO

python train.py --config-name _2_sokoban trainer.experiment_name=sokoban_ppo $USE_PPO