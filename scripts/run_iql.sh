export WANDB_ENTITY="astifer"
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_API_KEY="b89cebf2471bbc17859484af40db8b488c41af70"

cd /workspace/ml_3th_sem/algorithms

python3 iql.py \
  --actor_lr=3e-4 \
  --batch_size=256 \
  --beta=3.0 \
  --buffer_size=10000000 \
  --device="cuda" \
  --discount=0.99 \
  --env="halfcheetah-medium-expert-v2" \
  --eval_freq=5000 \
  --group="iql-halfcheetah-medium-expert-v2-multiseed-v0" \
  --iql_tau="0.7" \
  --load_model="" \
  --max_timesteps=1000000 \
  --n_episodes=10 \
  --name="IQL" \
  --project="offline_rl" \
  --critic_lr="3e-4" \
  --seed=0 \
  --tau=0.005 \
  --value_func_lr=3e-4