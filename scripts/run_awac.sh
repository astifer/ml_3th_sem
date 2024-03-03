export WANDB_ENTITY="astifer"
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_API_KEY="b89cebf2471bbc17859484af40db8b488c41af70"

cd /workspace/ml_3th_sem/algorithms

python3 awac.py \
  --awac_lambda=0.3333 \
  --batch_size=256 \
  --buffer_size=10000000 \
  --device="cuda" \
  --env_name="halfcheetah-medium-expert-v2" \
  --eval_frequency=1000 \
  --gamma=0.99 \
  --group="awac-halfcheetah-medium-expert-v2-multiseed-v0" \
  --hidden_dim=256 \
  --learning_rate=0.0003 \
  --n_test_episodes=10 \
  --num_train_ops=3000000 \
  --project="offline_rl" \
  --seed=42 \
  --tau=0.005 \
  --test_seed=69