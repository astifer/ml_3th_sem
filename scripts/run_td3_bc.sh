export WANDB_ENTITY="astifer"
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_API_KEY="b89cebf2471bbc17859484af40db8b488c41af70"

cd /workspace/ml_3th_sem/algorithms

python3 td3_bc.py \
  --alpha=2.5 \
  --batch_size=256 \
  --buffer_size=10000000 \
  --device="cuda" \
  --discount=0.99 \
  --env="hopper-medium-expert-v2" \
  --eval_freq=5000 \
  --expl_noise=0.1 \
  --group="td3-bc-hopper-medium-expert-v2-multiseed-v0" \
  --load_model="" \
  --max_timesteps=3000000 \
  --n_episodes=10 \
  --name="TD3-BC" \
  --noise_clip=0.5 \
  --policy_freq=2 \
  --policy_noise=0.2 \
  --project="offline_rl" \
  --seed=0 \
  --tau=0.005 \